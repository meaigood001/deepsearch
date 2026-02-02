import os
import logging
import sqlite3
import json
import secrets
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, Response, session
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
CORS(app, supports_credentials=True)

DB_PATH = Path("research_history.db")


def init_db():
    """Initialize SQLite database with users and research history schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Research history table with user_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            config TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_research_user_id 
        ON research_history(user_id, created_at DESC)
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def login_required(f):
    """Decorator to require authentication for routes."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Authentication required"}), 401
        return f(*args, **kwargs)

    return decorated_function


def get_current_user_id() -> Optional[int]:
    """Get current logged-in user ID."""
    return session.get("user_id")


def create_user(username: str, email: str, password: str) -> Optional[int]:
    """Create a new user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"User created: {username} (ID: {user_id})")
        return user_id
    except sqlite3.IntegrityError as e:
        logger.error(f"User creation failed: {e}")
        return None


def verify_user(username: str, password: str) -> Optional[Dict]:
    """Verify user credentials."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ?",
        (username,),
    )
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return {"id": user["id"], "username": user["username"], "email": user["email"]}
    return None


def save_research(
    query: str, config: Dict, result: Dict, user_id: int
) -> Optional[int]:
    """Save research to database for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO research_history (user_id, query, config, result)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, query, json.dumps(config), json.dumps(result)),
    )
    research_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Research saved with ID: {research_id} for user: {user_id}")
    return research_id


def get_research_history(user_id: int, limit: int = 20) -> list:
    """Get recent research history for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, query, created_at
        FROM research_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_research_by_id(research_id: int, user_id: int) -> Optional[Dict]:
    """Get research details by ID for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, query, config, result, created_at
        FROM research_history
        WHERE id = ? AND user_id = ?
    """,
        (research_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_research(research_id: int, user_id: int) -> bool:
    """Delete research from history for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM research_history WHERE id = ? AND user_id = ?",
        (research_id, user_id),
    )
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if deleted:
        logger.info(f"Research {research_id} deleted by user {user_id}")
    return deleted


# API Routes
@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "web_ui_modern.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files."""
    return send_from_directory("static", filename)


# Authentication Routes
@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user."""
    try:
        data = request.json
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not username or not email or not password:
            return jsonify({"success": False, "error": "All fields are required"}), 400

        if len(password) < 6:
            return jsonify(
                {"success": False, "error": "Password must be at least 6 characters"}
            ), 400

        user_id = create_user(username, email, password)
        if user_id:
            session["user_id"] = user_id
            session["username"] = username
            session.permanent = True
            return jsonify(
                {
                    "success": True,
                    "user": {"id": user_id, "username": username, "email": email},
                }
            )
        else:
            return jsonify(
                {"success": False, "error": "Username or email already exists"}
            ), 409
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Login user."""
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()

        if not username or not password:
            return jsonify(
                {"success": False, "error": "Username and password are required"}
            ), 400

        user = verify_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session.permanent = True
            return jsonify({"success": True, "user": user})
        else:
            return jsonify(
                {"success": False, "error": "Invalid username or password"}
            ), 401
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    """Logout user."""
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me", methods=["GET"])
@login_required
def get_current_user():
    """Get current logged-in user."""
    user_id = session.get("user_id")
    username = session.get("username")

    # Get user details from database
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,)
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        return jsonify({"success": True, "user": dict(user)})
    else:
        session.clear()
        return jsonify({"success": False, "error": "User not found"}), 404


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "authenticated": "user_id" in session,
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        }
    )


@app.route("/api/history", methods=["GET"])
@login_required
def get_history():
    """Get research history for current user."""
    try:
        user_id = get_current_user_id()
        assert user_id is not None, "User must be authenticated"
        limit = request.args.get("limit", default=20, type=int)
        history = get_research_history(user_id, limit)
        return jsonify({"success": True, "data": history})
    except Exception as e:
        logger.error(f"Error fetching history: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/research/<int:research_id>", methods=["GET"])
@login_required
def get_research(research_id):
    """Get research by ID for current user."""
    try:
        user_id = get_current_user_id()
        assert user_id is not None, "User must be authenticated"
        research = get_research_by_id(research_id, user_id)
        if research:
            research["result"] = json.loads(research["result"])
            research["config"] = json.loads(research["config"])
            return jsonify({"success": True, "data": research})
        return jsonify({"success": False, "error": "Research not found"}), 404
    except Exception as e:
        logger.error(f"Error fetching research: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/research/<int:research_id>", methods=["DELETE"])
@login_required
def delete_research_endpoint(research_id):
    """Delete research by ID for current user."""
    try:
        user_id = get_current_user_id()
        assert user_id is not None, "User must be authenticated"
        if delete_research(research_id, user_id):
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "Research not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting research: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/research/stream", methods=["POST"])
@login_required
def research_stream():
    """Stream research results for current user."""
    try:
        user_id = get_current_user_id()
        data = request.json
        query = data.get("query", "").strip()
        config = data.get("config", {})
        user_answers = data.get("user_answers", None)

        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400

        if not os.getenv("OPENAI_API_KEY"):
            return jsonify(
                {"success": False, "error": "OPENAI_API_KEY not configured"}
            ), 500

        def generate():
            last_update = None
            try:
                from research_agent_v4 import stream_research_agent

                for last_update in stream_research_agent(
                    query=query,
                    max_iterations=config.get("max_iterations", 3),
                    lang=config.get("lang", "en"),
                    char_limits=config.get(
                        "char_limits",
                        {
                            "background": 300,
                            "keyword_summary": 500,
                            "final_report": 2000,
                        },
                    ),
                    user_answers=user_answers,
                    allow_console_input=False,
                ):
                    yield f"data: {json.dumps(last_update)}\n\n"

                # Save research on completion
                if last_update and last_update.get("node") == "complete" and user_id:
                    research_id = save_research(
                        query, config, last_update["state"], user_id
                    )
                    yield f"data: {json.dumps({'node': 'saved', 'research_id': research_id})}\n\n"

            except Exception as e:
                logger.error(f"Research error: {e}", exc_info=True)
                yield f"data: {json.dumps({'node': 'error', 'error': str(e)})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5001))  # Changed from 5000 to 5001 to avoid conflicts
    app.run(host="0.0.0.0", port=port, debug=True)
