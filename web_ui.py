import os
import logging
import sqlite3
import json
from typing import Dict, Optional
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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


st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { padding-top: 2rem; }
    h1 {
        color: #5B21B6;
        font-weight: 700;
        border-bottom: 3px solid #7C3AED;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #5B21B6, #7C3AED, #A855F7);
    }
    .info-box {
        background: linear-gradient(135deg, rgba(91, 33, 182, 0.05) 0%, rgba(124, 58, 237, 0.05) 100%);
        border-left: 4px solid #7C3AED;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .node-box {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)


def sidebar_config() -> Dict:
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.subheader("🔬 Research Parameters")

    max_iterations = st.sidebar.slider(
        "Max Iterations",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of research iterations",
    )

    lang = st.sidebar.selectbox("Language", options=["English (en)", "Chinese (zh)"])

    with st.sidebar.expander("Advanced Settings", expanded=False):
        limit_background = st.number_input(
            "Background Summary", value=300, min_value=100, max_value=2000, step=50
        )
        limit_keyword = st.number_input(
            "Keyword Summary", value=500, min_value=200, max_value=3000, step=100
        )
        limit_final = st.number_input(
            "Final Report", value=2000, min_value=500, max_value=10000, step=500
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Research History")

    history = get_research_history(limit=10)
    if not history:
        st.sidebar.info("No research history yet.")
    else:
        for item in history:
            with st.sidebar.expander(f"🔎 {item['query'][:50]}...", expanded=False):
                st.caption(f"📅 {item['created_at']}")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button(
                        "📖 View", key=f"view_{item['id']}", use_container_width=True
                    ):
                        if "selected_research_id" not in st.session_state:
                            st.session_state.selected_research_id = None
                        st.session_state.selected_research_id = item["id"]
                        st.rerun()
                with col2:
                    if st.button(
                        "🗑️ Delete", key=f"delete_{item['id']}", use_container_width=True
                    ):
                        if delete_research(item["id"]):
                            st.rerun()

    return {
        "max_iterations": max_iterations,
        "lang": "en" if "en" in lang else "zh",
        "char_limits": {
            "background": limit_background,
            "keyword_summary": limit_keyword,
            "final_report": limit_final,
        },
    }


def display_results(result):
    if not result:
        return

    if result.get("background"):
        st.subheader("🔍 Background Research")
        st.markdown(result["background"])

    if result.get("keyword_history"):
        st.subheader("🎯 Search Keywords Used")
        keywords = result["keyword_history"]
        cols = st.columns(5)
        for idx, kw in enumerate(keywords):
            with cols[idx % 5]:
                st.markdown(
                    f'<span style="background: linear-gradient(90deg, #5B21B6, #7C3AED); color: white; padding: 8px 16px; border-radius: 20px; display: inline-block; margin: 4px 0;">{kw}</span>',
                    unsafe_allow_html=True,
                )

    if result.get("summaries"):
        st.subheader("📋 Search Summaries")
        llm_outputs = result.get("llm_outputs", {})
        for key in list(llm_outputs.keys()):
            if key.startswith("multi_search_iteration_"):
                iteration_data = llm_outputs[key]
                st.markdown(f"**Iteration {key.split('_')[-1]}:**")
                for item in iteration_data:
                    with st.expander(
                        f"🔎 {item.get('keyword', 'Unknown')}",
                        expanded=False,
                    ):
                        st.markdown(item.get("summary", ""))

    if result.get("gaps_found") is not None:
        gap_status = "⚠️ Gaps detected" if result["gaps_found"] else "✅ No gaps found"
        st.markdown(
            f'<div class="info-box"><strong>Gap Analysis:</strong> {gap_status}</div>',
            unsafe_allow_html=True,
        )

    if result.get("final_report"):
        st.subheader("📝 Final Report")
        st.markdown(result["final_report"])

        st.download_button(
            label="💾 Download Report",
            data=result["final_report"],
            file_name="research_report.md",
            mime="text/markdown",
        )


def main():
    st.title("🔍 Deep Research Agent")
    st.markdown(
        '<div class="info-box"><p><strong>Welcome!</strong> Ask any research question and get a comprehensive, multi-source analysis with real-time progress tracking.</p></div>',
        unsafe_allow_html=True,
    )

    config = sidebar_config()

    init_db()

    if "research_stage" not in st.session_state:
        st.session_state.research_stage = "idle"
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "selected_research_id" not in st.session_state:
        st.session_state.selected_research_id = None

    if st.session_state.selected_research_id:
        loaded_research = get_research_by_id(st.session_state.selected_research_id)
        if loaded_research:
            st.session_state.current_result = json.loads(loaded_research["result"])
            st.session_state.research_stage = "viewing"
            st.session_state.last_query = loaded_research["query"]
            st.session_state.selected_research_id = None

    st.subheader("📝 Your Research Question")

    query = st.text_area(
        "Enter your research question...",
        placeholder="e.g., What are the latest developments in AI safety?",
        height=100,
        label_visibility="collapsed",
    )

    if (
        st.session_state.research_stage != "viewing"
        and query != st.session_state.last_query
    ):
        st.session_state.research_stage = "idle"
        st.session_state.current_result = None
        st.session_state.last_query = query

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.research_stage == "idle":
            run_button = st.button(
                "🚀 Start Research", type="primary", use_container_width=True
            )
        else:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.research_stage = "idle"
                st.session_state.current_result = None
                st.rerun()
            run_button = False

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "⚠️ **Missing API Key**: Please set `OPENAI_API_KEY` in your `.env` file."
        )
        return

    if not os.getenv("TAVILY_API_KEY"):
        st.warning(
            "⚠️ **Missing Tavily API**: Please set `TAVILY_API_KEY` in your `.env` file for web search."
        )

    if run_button and query:
        if not query.strip():
            st.error("❌ Please enter a research question.")
        else:
            with st.status("🚀 Running Research...", expanded=True) as status:
                st.write("📊 Initializing...")
                try:
                    from research_agent_v4 import run_research_agent

                    st.write("🔎 Analyzing query for clarification...")
                    result = run_research_agent(
                        query=query,
                        max_iterations=config["max_iterations"],
                        lang=config["lang"],
                        char_limits=config["char_limits"],
                        allow_console_input=False,
                    )

                    st.session_state.current_result = result
                    if result and result.get("clarification_needed"):
                        st.session_state.research_stage = "clarifying"
                        status.update(label="❓ Clarification Needed", state="complete")
                    else:
                        st.session_state.research_stage = "completed"
                        status.update(label="✅ Research Completed", state="complete")
                        save_research(query, config, result)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Research error: {e}", exc_info=True)

    if st.session_state.research_stage == "clarifying":
        result = st.session_state.current_result
        if result:
            questions = result.get("clarification_questions", [])

            st.markdown("---")
            st.subheader("❓ Clarification Needed")
            st.info("To conduct more accurate research, please answer the following:")

            with st.form("clarification_form"):
                user_answers = {}
                for idx, q in enumerate(questions, 1):
                    user_answers[str(idx)] = st.text_input(
                        f"Q{idx}: {q}", key=f"q_{idx}"
                    )

                if st.form_submit_button("🚀 Continue Research", type="primary"):
                    with st.status(
                        "🔄 Continuing Research...", expanded=True
                    ) as status:
                        try:
                            from research_agent_v4 import run_research_agent

                            st.write("Processing clarifications and searching...")

                            final_result = run_research_agent(
                                query=query,
                                max_iterations=config["max_iterations"],
                                lang=config["lang"],
                                char_limits=config["char_limits"],
                                user_answers=user_answers,
                                allow_console_input=False,
                            )
                            st.session_state.current_result = final_result
                            st.session_state.research_stage = "completed"
                            status.update(
                                label="✅ Research Completed", state="complete"
                            )
                            st.rerun()
                        except Exception as e:
                            import traceback

                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc(), language="python")
                            logger.error(f"Research error: {e}", exc_info=True)

    if st.session_state.research_stage == "viewing":
        st.markdown("---")
        st.info("📜 Viewing past research from history")
        display_results(st.session_state.current_result)
        if st.button("🔄 New Research", use_container_width=True):
            st.session_state.research_stage = "idle"
            st.session_state.current_result = None
            st.session_state.last_query = ""
            st.rerun()

    if st.session_state.research_stage == "completed":
        st.markdown("---")
        display_results(st.session_state.current_result)


if __name__ == "__main__":
    main()
