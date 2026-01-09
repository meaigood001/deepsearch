import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated, Dict
import operator
import datetime
import logging
import json
import re
import argparse
from pathlib import Path
from pydantic import BaseModel, Field

from config import load_model_config
from llm_factory import create_llm_instances

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_time_awareness_instruction(current_time: str, lang: str = "en") -> str:
    logger.debug(
        f"Generating time awareness instruction for time: {current_time}, language: {lang}"
    )
    instructions = {
        "en": f"""
⏰ Critical Time Information:
- Current real time is: {current_time}
- This is the actual current date and time. Do not question or "correct" this date.
- Your training data cutoff date is outdated. Please accept the provided current time as the real basis.
- When searching for information, use the year from the current time, not the year from your training data.
- If the user asks about recent events or "latest" information, base it on the current time provided above.
- If the user's query mentions a specific year or time period, include it in your search queries.
""",
        "zh": f"""
⏰ 关键时间信息:
- 当前真实时间是: {current_time}
- 这是实际的当前日期和时间。请不要质疑或"纠正"这个日期。
- 你的训练数据截止日期已经过时。请接受提供的当前时间作为真实依据。
- 在搜索信息时，使用当前时间中的年份，而不是你训练数据中的年份。
- 如果用户询问最近的事件或"最新"信息，请基于上面提供的当前时间。
- 如果用户的查询提到了特定的年份或时间段，请在搜索查询中包含它。
""",
    }
    result = instructions.get(lang, instructions["en"])
    logger.debug(f"Time awareness instruction generated for language: {lang}")
    return result


def get_current_time() -> str:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug(f"Current time retrieved: {current_time}")
    return current_time


# Define the state
class ResearchState(TypedDict):
    query: str
    original_query: str
    background: str
    confirmed: bool
    keywords: list
    summaries: Annotated[list, operator.add]
    gaps_found: bool
    final_report: str
    current_time: str
    iteration: int
    max_iterations: int
    keyword_history: list
    lang: str


# Define tools
@tool
def web_search(query: str) -> str:
    """Search the web for information related to the query."""
    search = TavilySearch(max_results=5)
    results = search.invoke({"query": query})
    return str(results)


@tool
def analyze_insight(info: str) -> str:
    """Analyze and extract key insights from information."""
    return f"Key insights from: {info}"


class KeywordsResponse(BaseModel):
    keywords: list[str] = Field(
        description="List of 3-5 search keywords, no explanations"
    )


# Define nodes - LLM instances will be initialized after CLI args are parsed
llm_instances: Dict[str, ChatOpenAI] = {}


def background_search_node(state: ResearchState):
    logger.info("=== BACKGROUND SEARCH NODE STARTED ===")
    query = state["query"]
    logger.info(f"Background search query: {query}")

    search_tool = web_search
    logger.debug(f"Invoking web search tool for query: {query}")
    search_results = search_tool.invoke(query)

    logger.info(
        f"Background search completed. Results length: {len(str(search_results))} characters"
    )
    logger.debug(f"Background search results preview: {str(search_results)[:200]}...")

    time_instruction = get_time_awareness_instruction(
        state["current_time"], state["lang"]
    )

    prompt = ChatPromptTemplate.from_template(
        f"{time_instruction}\n"
        f"🎯 USER'S ORIGINAL REQUEST: {state['original_query']}\n"
        f"💡 IMPORTANT: The background summary should provide essential context and foundational knowledge about the topic.\n"
        f"Summarize the following search results for the query '{{query}}' into a comprehensive background summary:\n"
        f"{{search_results}}"
    )

    chain = prompt | llm_instances["background_search"] | StrOutputParser()

    logger.debug("Invoking LLM to generate background summary")
    background = chain.invoke({"query": query, "search_results": search_results})
    logger.info(f"Background summary generated: {len(background)} characters")

    return {
        "background": background,
        "confirmed": True,
    }


def generate_keywords_node(state: ResearchState):
    logger.info("=== GENERATE KEYWORDS NODE STARTED ===")
    background = state["background"]
    query = state["query"]
    original_query = state["original_query"]
    iteration = state["iteration"]
    keyword_history = state["keyword_history"]
    max_iterations = state["max_iterations"]

    logger.info(
        f"Generating keywords for query: {query} (Iteration {iteration}/{max_iterations})"
    )
    logger.debug(f"Original query: {original_query}")
    logger.debug(f"Background context length: {len(str(background))} characters")
    logger.debug(f"Keyword history: {keyword_history}")

    time_instruction = get_time_awareness_instruction(
        state["current_time"], state["lang"]
    )

    prompt = ChatPromptTemplate.from_template(
        f"{time_instruction}\n"
        f"🎯 USER'S ORIGINAL REQUEST: {original_query}\n"
        f"💡 IMPORTANT: Your search should help answer their specific question, not just collect general information about the paragraph topic.\n"
        f"📋 INSTRUCTIONS:\n"
        f"- Generate 3-5 search keywords for google search in-depth research on: {{query}}\n"
        f"- Based on background information: {{background}}\n"
        f"- Keywords must be different from previous searches: {keyword_history}\n"
        f"- Return as valid JSON with key 'keywords' containing array of 3-5 strings\n"
        f"- Do NOT include markdown code blocks, just the JSON\n"
        f"- Do NOT include explanations or numbered lists\n"
    )

    chain = prompt | llm_instances["generate_keywords"] | StrOutputParser()
    logger.debug("Invoking LLM to generate keywords")
    llm_output = chain.invoke({"background": background, "query": query})

    logger.debug(f"Raw LLM output: {llm_output}")

    keywords = []
    try:
        json_match = re.search(r"\{[\s\S]*?\}", llm_output)
        if json_match:
            json_str = json_match.group(0)
            logger.debug(f"Extracted JSON: {json_str}")
            parsed = json.loads(json_str)
            keywords = parsed.get("keywords", [])
        else:
            logger.warning("No JSON found, falling back to comma extraction")
            if "," in llm_output:
                keywords = [
                    kw.strip()
                    for kw in llm_output.split(",")
                    if kw.strip() and len(kw.strip()) > 2
                ]
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}, falling back to comma extraction")
        if "," in llm_output:
            keywords = [
                kw.strip()
                for kw in llm_output.split(",")
                if kw.strip() and len(kw.strip()) > 2
            ]
    except Exception as e:
        logger.warning(f"Error parsing keywords: {e}")
        if "," in llm_output:
            keywords = [
                kw.strip()
                for kw in llm_output.split(",")
                if kw.strip() and len(kw.strip()) > 2
            ]

    logger.debug(f"Parsed keywords: {keywords}")

    keywords = [kw for kw in keywords if kw and len(kw) > 2]
    keywords = list(dict.fromkeys(keywords))
    keywords = [
        kw for kw in keywords if kw.lower() not in [h.lower() for h in keyword_history]
    ]
    keywords = keywords[:5]

    new_keyword_history = keyword_history + keywords
    new_iteration = iteration + 1

    logger.info(f"Generated {len(keywords)} unique new keywords: {keywords}")
    logger.info(f"Updated keyword history: {new_keyword_history}")
    logger.info(f"Iteration counter updated: {new_iteration}/{max_iterations}")

    return {
        "keywords": keywords,
        "iteration": new_iteration,
        "keyword_history": new_keyword_history,
    }


def multi_search_node(state: ResearchState):
    logger.info("=== MULTI SEARCH NODE STARTED ===")
    keywords = state["keywords"]
    summaries = []
    original_query = state["original_query"]

    logger.info(f"Starting multi-search for {len(keywords)} keywords: {keywords}")

    for idx, kw in enumerate(keywords, 1):
        logger.info(f"Processing keyword {idx}/{len(keywords)}: {kw}")

        search_tool = web_search
        logger.debug(f"Searching for keyword: {kw}")
        result = search_tool.invoke(kw)
        logger.debug(f"Search results for '{kw}': {len(str(result))} characters")

        prompt = ChatPromptTemplate.from_template(
            f"{get_time_awareness_instruction(state['current_time'], state['lang'])}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: The paragraph must help answer the user's original research question. Focus on information relevant to their specific inquiry.\nSummarize the following search results for keyword '{{kw}}': {{result}}"
        )
        chain = prompt | llm_instances["multi_search"] | StrOutputParser()

        logger.debug(f"Summarizing results for keyword: {kw}")
        summary = chain.invoke({"kw": kw, "result": result})
        logger.info(f"Generated summary for '{kw}': {len(summary)} characters")
        summaries.append(summary)

    logger.info(f"Multi-search completed. Generated {len(summaries)} summaries")
    return {"summaries": summaries}


def check_gaps_node(state: ResearchState):
    logger.info("=== CHECK GAPS NODE STARTED ===")
    summaries = state["summaries"]
    query = state["query"]
    original_query = state["original_query"]

    logger.info(f"Checking gaps for {len(summaries)} summaries against query: {query}")
    logger.debug(f"Original query context: {original_query}")

    prompt = ChatPromptTemplate.from_template(
        f"{get_time_awareness_instruction(state['current_time'], state['lang'])}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: Reflect on whether the current paragraphs sufficiently address the user's original research question. Identify deficiencies, especially in answering the query.\nReview these summaries: {{summaries}}\nAgainst the query: {{query}}\nAre there gaps? Answer 'yes' or 'no'."
    )
    chain = prompt | llm_instances["check_gaps"] | StrOutputParser()

    logger.debug("Invoking LLM to check for gaps")
    gaps = chain.invoke({"summaries": "\n".join(summaries), "query": query})
    logger.debug(f"Gap analysis result: {gaps}")

    gaps_found = "yes" in gaps.lower()
    logger.info(f"Gap analysis complete. Gaps found: {gaps_found}")

    return {"gaps_found": gaps_found}


def synthesize_node(state: ResearchState):
    logger.info("=== SYNTHESIZE NODE STARTED ===")
    summaries = state["summaries"]
    query = state["query"]
    original_query = state["original_query"]

    logger.info(f"Synthesizing {len(summaries)} summaries into final report")
    logger.debug(f"Target query: {query}")
    logger.debug(f"Original query context: {original_query}")
    logger.debug(
        f"Total summary content length: {sum(len(s) for s in summaries)} characters"
    )

    prompt = ChatPromptTemplate.from_template(
        f"{get_time_awareness_instruction(state['current_time'], state['lang'])}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: Ensure final report comprehensively answers user's original research question. Conclude by summarizing how report addresses their specific inquiry.\nSynthesize these summaries into a comprehensive report for the query: {{query}}\nSummaries: {{summaries}}"
    )
    chain = prompt | llm_instances["synthesize"] | StrOutputParser()

    logger.debug("Invoking LLM to synthesize final report")
    report = chain.invoke({"query": query, "summaries": "\n".join(summaries)})
    logger.info(f"Final report generated: {len(report)} characters")

    return {"final_report": report}


# Build the graph
builder = StateGraph(ResearchState)
builder.add_node("background_search", background_search_node)
builder.add_node("generate_keywords", generate_keywords_node)
builder.add_node("multi_search", multi_search_node)
builder.add_node("check_gaps", check_gaps_node)
builder.add_node("synthesize", synthesize_node)

builder.add_edge(START, "background_search")
builder.add_edge("background_search", "generate_keywords")
builder.add_edge("generate_keywords", "multi_search")
builder.add_edge("multi_search", "check_gaps")


# Conditional edge: if gaps_found, loop back to generate_keywords; else to synthesize
def route_after_check(state: ResearchState):
    gaps_found = state["gaps_found"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    if iteration >= max_iterations:
        logger.info(
            f"Routing decision: Max iterations ({max_iterations}) reached. Forcing synthesize."
        )
        return "synthesize"

    next_node = "generate_keywords" if gaps_found else "synthesize"
    logger.info(
        f"Routing decision: Gaps found = {gaps_found}, Iteration {iteration}/{max_iterations}, Next node = {next_node}"
    )
    return next_node


builder.add_conditional_edges("check_gaps", route_after_check)
builder.add_edge("synthesize", END)

graph = builder.compile()


# Function to run the agent
def run_research_agent(query: str, max_iterations: int = 3, lang: str = "en"):
    logger.info("=" * 50)
    logger.info(f"RESEARCH AGENT STARTED")
    logger.info(f"Query: {query}")
    logger.info("=" * 50)

    current_time = get_current_time()
    logger.info(f"Current time: {current_time}")

    initial_state = {
        "query": query,
        "original_query": query,
        "background": "",
        "confirmed": False,
        "keywords": [],
        "summaries": [],
        "gaps_found": False,
        "final_report": "",
        "current_time": current_time,
        "iteration": 0,
        "max_iterations": max_iterations,
        "keyword_history": [],
        "lang": lang,
    }

    logger.debug("Initial state prepared, starting graph execution")
    result = graph.invoke(initial_state)

    logger.info("=" * 50)
    logger.info(f"RESEARCH AGENT COMPLETED")
    logger.info(f"Final report length: {len(result['final_report'])} characters")
    logger.info("=" * 50)

    return result["final_report"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="deepsearch",
        description="Deep Research Agent - AI-powered multi-iteration research tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: %(prog)s 'What are the latest developments in AI safety?' -o report.md",
    )

    # Positional argument (required)
    parser.add_argument(
        "query",
        type=str,
        help="Research question to investigate",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (auto-generated if not specified)",
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="markdown",
        choices=["markdown", "txt", "json"],
        help="Output format",
    )

    # Research options
    parser.add_argument(
        "--max-iterations",
        "-i",
        type=int,
        default=3,
        help="Maximum number of research iterations",
    )

    # Global model configuration
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Global model name for all nodes (overrides MODEL_NAME env var)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Global API base URL for all nodes (overrides OPENAI_BASE_URL env var)",
    )

    # Node-specific model configuration
    model_group = parser.add_argument_group(
        "Node-specific models",
        "Configure different models for each research agent node",
    )

    model_group.add_argument(
        "--model-generate-keywords",
        type=str,
        default=None,
        help="Model for generate_keywords node",
    )
    model_group.add_argument(
        "--model-multi-search",
        type=str,
        default=None,
        help="Model for multi_search node",
    )
    model_group.add_argument(
        "--model-check-gaps",
        type=str,
        default=None,
        help="Model for check_gaps node",
    )
    model_group.add_argument(
        "--model-synthesize",
        type=str,
        default=None,
        help="Model for synthesize node",
    )

    # Node-specific base URL configuration
    base_url_group = parser.add_argument_group(
        "Node-specific base URLs",
        "Configure different API base URLs for each research agent node",
    )

    base_url_group.add_argument(
        "--base-url-generate-keywords",
        type=str,
        default=None,
        help="API base URL for generate_keywords node",
    )
    base_url_group.add_argument(
        "--base-url-multi-search",
        type=str,
        default=None,
        help="API base URL for multi_search node",
    )
    base_url_group.add_argument(
        "--base-url-check-gaps",
        type=str,
        default=None,
        help="API base URL for check_gaps node",
    )
    base_url_group.add_argument(
        "--base-url-synthesize",
        type=str,
        default=None,
        help="API base URL for synthesize node",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Language for time-aware prompts (en/zh)",
    )

    # Logging options
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Enable quiet mode (WARNING level)",
    )

    args = parser.parse_args()

    # Validate query
    if len(args.query) < 3:
        parser.error("Query must be at least 3 characters long")

    if len(args.query) > 1000:
        parser.error("Query too long (max 1000 characters)")

    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        parser.error(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment.\n"
            "See .env.example for required variables."
        )

    # Set log level
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"

    logging.getLogger().setLevel(getattr(logging, log_level))
    logger.info(f"Log level set to: {log_level}")

    # Build CLI overrides for model configuration
    cli_overrides = {}

    # Global overrides
    if args.model or args.base_url:
        for node in ["generate_keywords", "multi_search", "check_gaps", "synthesize"]:
            cli_overrides[node] = {}
            if args.model:
                cli_overrides[node]["model"] = args.model
            if args.base_url:
                cli_overrides[node]["base_url"] = args.base_url

    # Node-specific overrides
    if args.model_generate_keywords:
        cli_overrides.setdefault("generate_keywords", {})["model"] = (
            args.model_generate_keywords
        )
    if args.base_url_generate_keywords:
        cli_overrides.setdefault("generate_keywords", {})["base_url"] = (
            args.base_url_generate_keywords
        )

    if args.model_multi_search:
        cli_overrides.setdefault("multi_search", {})["model"] = args.model_multi_search
    if args.base_url_multi_search:
        cli_overrides.setdefault("multi_search", {})["base_url"] = (
            args.base_url_multi_search
        )

    if args.model_check_gaps:
        cli_overrides.setdefault("check_gaps", {})["model"] = args.model_check_gaps
    if args.base_url_check_gaps:
        cli_overrides.setdefault("check_gaps", {})["base_url"] = (
            args.base_url_check_gaps
        )

    if args.model_synthesize:
        cli_overrides.setdefault("synthesize", {})["model"] = args.model_synthesize
    if args.base_url_synthesize:
        cli_overrides.setdefault("synthesize", {})["base_url"] = (
            args.base_url_synthesize
        )

    # Initialize LLM instances with configuration
    model_config = load_model_config(cli_overrides)
    llm_instances_global = create_llm_instances(model_config)
    llm_instances.update(llm_instances_global)

    # Generate output filename if not provided
    output_file = args.output
    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r"[^\w\s-]", "", args.query)[:30].strip()
        safe_query = re.sub(r"\s+", "_", safe_query)
        ext = "md" if args.format == "markdown" else args.format.lower()
        output_file = f"output/{safe_query}_{timestamp}.{ext}"

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {output_path.parent}")

    # Run research agent
    logger.info(f"Running research agent with query: {args.query}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Output file: {output_file}")

    try:
        report = run_research_agent(
            args.query, max_iterations=args.max_iterations, lang=args.lang
        )
    except Exception as e:
        logger.error(f"Research agent failed: {e}", exc_info=True)
        # Save partial results if output file specified
        if args.output:
            import traceback

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Research Failed\n\nError: {e}\n\n{traceback.format_exc()}")
        raise

    # Write output file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            if args.format == "json":
                json.dump(
                    {
                        "query": args.query,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "report": report,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            else:
                f.write(report)

        logger.info(f"✅ Report saved to: {output_file}")
    except IOError as e:
        logger.error(f"❌ Failed to write output file: {e}")
        raise

    # Also print to console for immediate feedback
    print(report)
    logger.info("Research agent execution finished")
