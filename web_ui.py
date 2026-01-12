"""
Streamlit Web UI for Deep Research Agent
"""

import os
import logging
from typing import Dict

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
    .css-1d391kg {
        background: linear-gradient(180deg, #5B21B6 0%, #7C3AED 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)


def sidebar_config() -> Dict:
    """Render sidebar configuration and return config dict."""
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

    return {
        "max_iterations": max_iterations,
        "lang": "en" if "en" in lang else "zh",
        "char_limits": {
            "background": limit_background,
            "keyword_summary": limit_keyword,
            "final_report": limit_final,
        },
    }


def main():
    """Main Streamlit application."""
    st.title("🔍 Deep Research Agent")
    st.markdown(
        """
    <div class="info-box">
        <p><strong>Welcome!</strong> Ask any research question and get a comprehensive,
        multi-source analysis with real-time progress tracking.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    config = sidebar_config()

    st.subheader("📝 Your Research Question")

    query = st.text_area(
        "Enter your research question...",
        placeholder="e.g., What are the latest developments in AI safety?",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "🚀 Start Research", type="primary", use_container_width=True
        )

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
            return

        progress_container = st.container()
        results_container = st.container()

        with progress_container:
            st.info("🔄 Research in progress...")
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            status_text.text("📊 Loading...")
            progress_bar.progress(10, text="Config loaded")

            try:
                from research_agent_v4 import run_research_agent

                status_text.text("🚀 Starting research...")
                progress_bar.progress(20, text="Research started")

                result = run_research_agent(
                    query=query,
                    max_iterations=config["max_iterations"],
                    lang=config["lang"],
                    char_limits=config["char_limits"],
                )

                progress_bar.progress(100, text="Research completed!")
                status_text.success("✅ Research completed successfully!")

                with results_container:
                    st.markdown("---")

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
                                    f'<span style="background: linear-gradient(90deg, #5B21B6, #7C3AED); color: white; padding: 8px 16px; border-radius: 20px;">{kw}</span>',
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
                        gap_status = (
                            "⚠️ Gaps detected"
                            if result["gaps_found"]
                            else "✅ No gaps found"
                        )
                        st.markdown(
                            f"""
                        <div class="info-box">
                            <strong>Gap Analysis:</strong> {gap_status}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    if result.get("final_report"):
                        st.subheader("📝 Final Report")
                        st.markdown(result["final_report"])

                        st.download_button(
                            label="💾 Download Report",
                            data=result["final_report"],
                            file_name=f"research_report.md",
                            mime="text/markdown",
                        )

            except Exception as e:
                error_msg = f"❌ Error during research: {str(e)}"
                st.error(error_msg)
                logger.error(f"Research execution error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
