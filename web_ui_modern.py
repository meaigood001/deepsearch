import os
import logging
import sqlite3
import json
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

DB_PATH = Path("research_history.db")


def init_db():
    """Initialize SQLite database with research history schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            config TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def save_research(query: str, config: Dict, result: Dict) -> Optional[int]:
    """Save research to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO research_history (query, config, result)
        VALUES (?, ?, ?)
        """,
        (query, json.dumps(config), json.dumps(result)),
    )
    research_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Research saved with ID: {research_id}")
    return research_id


def get_research_history(limit: int = 20) -> list:
    """Get recent research history."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, query, created_at
        FROM research_history
        ORDER BY created_at DESC
        LIMIT ?
    """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_research_by_id(research_id: int) -> Optional[Dict]:
    """Get research details by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, query, config, result, created_at
        FROM research_history
        WHERE id = ?
    """,
        (research_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_research(research_id: int) -> bool:
    """Delete research from history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM research_history WHERE id = ?", (research_id,))
    conn.commit()
    conn.close()
    logger.info(f"Research {research_id} deleted")
    return True


# API Routes
@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'web_ui_modern.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY"))
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get research history."""
    try:
        limit = request.args.get('limit', default=20, type=int)
        history = get_research_history(limit)
        return jsonify({"success": True, "data": history})
    except Exception as e:
        logger.error(f"Error fetching history: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/research/<int:research_id>', methods=['GET'])
def get_research(research_id):
    """Get research by ID."""
    try:
        research = get_research_by_id(research_id)
        if research:
            research['result'] = json.loads(research['result'])
            research['config'] = json.loads(research['config'])
            return jsonify({"success": True, "data": research})
        return jsonify({"success": False, "error": "Research not found"}), 404
    except Exception as e:
        logger.error(f"Error fetching research: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/research/<int:research_id>', methods=['DELETE'])
def delete_research_endpoint(research_id):
    """Delete research by ID."""
    try:
        delete_research(research_id)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting research: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/research/stream', methods=['POST'])
def research_stream():
    """Stream research results."""
    try:
        data = request.json
        query = data.get('query', '').strip()
        config = data.get('config', {})
        user_answers = data.get('user_answers', None)

        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400

        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({
                "success": False,
                "error": "OPENAI_API_KEY not configured"
            }), 500

        def generate():
            try:
                from research_agent_v4 import stream_research_agent

                for update in stream_research_agent(
                    query=query,
                    max_iterations=config.get('max_iterations', 3),
                    lang=config.get('lang', 'en'),
                    char_limits=config.get('char_limits', {
                        'background': 300,
                        'keyword_summary': 500,
                        'final_report': 2000
                    }),
                    user_answers=user_answers,
                    allow_console_input=False,
                ):
                    yield f"data: {json.dumps(update)}\n\n"

                # Save research on completion
                if update.get('node') == 'complete':
                    research_id = save_research(query, config, update['state'])
                    yield f"data: {json.dumps({'node': 'saved', 'research_id': research_id})}\n\n"

            except Exception as e:
                logger.error(f"Research error: {e}", exc_info=True)
                yield f"data: {json.dumps({'node': 'error', 'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    init_db()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
