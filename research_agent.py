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
    llm_outputs: dict
    html_report: str


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
    logger.debug(f"Background summary output: {background}")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs["background_search"] = background

    return {
        "background": background,
        "confirmed": True,
        "llm_outputs": llm_outputs,
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

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs[f"generate_keywords_iteration_{iteration}"] = {
        "raw_output": llm_output,
        "parsed_keywords": keywords,
    }

    return {
        "keywords": keywords,
        "iteration": new_iteration,
        "keyword_history": new_keyword_history,
        "llm_outputs": llm_outputs,
    }


def multi_search_node(state: ResearchState):
    logger.info("=== MULTI SEARCH NODE STARTED ===")
    keywords = state["keywords"]
    summaries = []
    original_query = state["original_query"]

    logger.info(f"Starting multi-search for {len(keywords)} keywords: {keywords}")

    multi_search_outputs = []
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
        logger.debug(f"Summary output for '{kw}': {summary}")
        summaries.append(summary)
        multi_search_outputs.append({"keyword": kw, "summary": summary})

    logger.info(f"Multi-search completed. Generated {len(summaries)} summaries")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs[f"multi_search_iteration_{state['iteration'] - 1}"] = (
        multi_search_outputs
    )

    return {"summaries": summaries, "llm_outputs": llm_outputs}


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

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs[f"check_gaps_iteration_{state['iteration'] - 1}"] = {
        "raw_output": gaps,
        "gaps_found": gaps_found,
    }

    return {"gaps_found": gaps_found, "llm_outputs": llm_outputs}


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
    logger.debug(f"Final report output: {report}")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs["synthesize"] = report

    return {"final_report": report, "llm_outputs": llm_outputs}


def generate_html_node(state: ResearchState):
    logger.info("=== GENERATE HTML NODE STARTED ===")

    original_query = state["original_query"]
    current_time = state["current_time"]
    llm_outputs = state.get("llm_outputs", {})

    logger.info("Generating HTML report")

    def escape_js_string(text):
        if not text:
            return ""
        text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        return text

    content_data = {}

    if "background_search" in llm_outputs:
        content_data["background"] = escape_js_string(llm_outputs["background_search"])

    iterations = set()
    for key in llm_outputs.keys():
        if key.startswith("generate_keywords_iteration_"):
            iteration = key.split("_")[-1]
            iterations.add(iteration)
            if "keywords" not in content_data:
                content_data["keywords"] = {}
            if iteration not in content_data["keywords"]:
                content_data["keywords"][iteration] = {"keywords": []}
            for k, v in llm_outputs.items():
                if k == f"generate_keywords_iteration_{iteration}":
                    content_data["keywords"][iteration]["keywords"] = v.get(
                        "parsed_keywords", []
                    )

    for key, value in llm_outputs.items():
        if key.startswith("multi_search_iteration_"):
            iteration = key.split("_")[-1]
            if "summaries" not in content_data:
                content_data["summaries"] = {}
            content_data["summaries"][iteration] = [
                {
                    "keyword": item["keyword"],
                    "summary": escape_js_string(item["summary"]),
                }
                for item in value
            ]

    for key, value in llm_outputs.items():
        if key.startswith("check_gaps_iteration_"):
            iteration = key.split("_")[-1]
            if "gaps" not in content_data:
                content_data["gaps"] = {}
            content_data["gaps"][iteration] = {
                "gaps_found": value.get("gaps_found", False),
                "raw_output": escape_js_string(value.get("raw_output", "")),
            }

    if "synthesize" in llm_outputs:
        content_data["final"] = escape_js_string(llm_outputs["synthesize"])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {original_query}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --primary-gradient: linear-gradient(135deg, #5B21B6 0%, #7C3AED 50%, #A855F7 100%);
            --secondary-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            --accent-purple: #7C3AED;
            --accent-violet: #8B5CF6;
            --accent-pink: #EC4899;
            --text-primary: #1F2937;
            --text-secondary: #6B7280;
            --text-light: #9CA3AF;
            --bg-primary: #FFFFFF;
            --bg-secondary: #F9FAFB;
            --bg-tertiary: #F3F4F6;
            --border-color: #E5E7EB;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.75;
            color: var(--text-primary);
            background: linear-gradient(135deg, #5B21B6 0%, #7C3AED 25%, #8B5CF6 50%, #A855F7 75%, #EC4899 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            padding: 20px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}

        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        .container {{
            max-width: 1200px;
            margin: 20px auto;
            background: var(--bg-primary);
            border-radius: 20px;
            box-shadow: var(--shadow-2xl);
            overflow: hidden;
            animation: fadeInUp 0.6s ease-out;
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .header {{
            background: var(--primary-gradient);
            color: white;
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 40px 40px;
            animation: patternMove 60s linear infinite;
        }}

        @keyframes patternMove {{
            0% {{ transform: translate(0, 0); }}
            100% {{ transform: translate(40px, 40px); }}
        }}

        .header h1 {{
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 3.5em;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            position: relative;
            letter-spacing: -0.02em;
        }}

        .header .meta {{
            font-size: 1em;
            opacity: 0.95;
            line-height: 1.8;
            font-weight: 400;
            position: relative;
        }}

        .header .meta p {{
            margin: 8px 0;
        }}

        .header .meta strong {{
            font-weight: 600;
            opacity: 1;
        }}

        .content {{
            padding: 50px;
            background: var(--bg-secondary);
        }}

        .section {{
            margin-bottom: 45px;
            background: var(--bg-primary);
            padding: 35px;
            border-radius: 16px;
            border-left: 5px solid var(--accent-purple);
            box-shadow: var(--shadow-md);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .section:hover {{
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }}

        .section::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(124, 58, 237, 0.05) 0%, transparent 70%);
            pointer-events: none;
        }}

        .section h2 {{
            color: var(--accent-purple);
            margin-bottom: 25px;
            font-size: 2em;
            font-weight: 700;
            border-bottom: 3px solid var(--border-color);
            padding-bottom: 15px;
            font-family: 'Playfair Display', Georgia, serif;
            position: relative;
        }}

        .section h2::after {{
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 60px;
            height: 3px;
            background: var(--secondary-gradient);
            border-radius: 3px;
        }}

        .section h3 {{
            color: var(--accent-violet);
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.5em;
            font-weight: 600;
        }}

        .section p {{
            margin-bottom: 20px;
            line-height: 1.85;
            color: var(--text-primary);
        }}

        .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 20px 0;
        }}

        .keyword-tag {{
            background: var(--secondary-gradient);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.95em;
            font-weight: 500;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .keyword-tag:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: var(--shadow-md);
        }}

        .keyword-tag::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }}

        .keyword-tag:hover::before {{
            left: 100%;
        }}

        .iteration-badge {{
            display: inline-block;
            background: var(--accent-violet);
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-bottom: 15px;
            box-shadow: var(--shadow-sm);
            letter-spacing: 0.5px;
        }}

        .final-report {{
            background: var(--primary-gradient);
            color: white;
            border-left: none;
            position: relative;
        }}

        .final-report::before {{
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        }}

        .final-report h2 {{
            color: white;
            border-bottom: 3px solid rgba(255,255,255,0.25);
            font-family: 'Playfair Display', Georgia, serif;
        }}

        .final-report h2::after {{
            background: linear-gradient(90deg, #fff, rgba(255,255,255,0.5));
        }}

        .final-report .markdown-content {{
            color: rgba(255,255,255,0.95);
        }}

        .final-report .markdown-content strong {{
            color: rgba(255,255,255,1);
            font-weight: 700;
        }}

        .footer {{
            text-align: center;
            padding: 30px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-size: 0.95em;
            font-weight: 400;
            border-top: 1px solid var(--border-color);
        }}

        .footer p {{
            margin: 5px 0;
        }}

        .markdown-content {{
            animation: fadeIn 0.5s ease-in;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .markdown-content h1 {{
            color: var(--accent-purple);
            margin: 35px 0 25px;
            font-size: 2.25em;
            border-bottom: 3px solid var(--border-color);
            padding-bottom: 15px;
            font-family: 'Playfair Display', Georgia, serif;
            font-weight: 700;
        }}

        .markdown-content h2 {{
            color: var(--accent-violet);
            margin: 30px 0 20px;
            font-size: 1.75em;
            font-weight: 600;
        }}

        .markdown-content h3 {{
            color: var(--accent-purple);
            margin: 25px 0 15px;
            font-size: 1.4em;
            font-weight: 600;
        }}

        .markdown-content p {{
            margin-bottom: 20px;
            line-height: 1.85;
            color: var(--text-primary);
        }}

        .markdown-content ul, .markdown-content ol {{
            margin-left: 25px;
            margin-bottom: 20px;
        }}

        .markdown-content li {{
            margin-bottom: 10px;
            line-height: 1.75;
        }}

        .markdown-content strong {{
            color: var(--accent-purple);
            font-weight: 700;
        }}

        .markdown-content em {{
            color: var(--accent-violet);
            font-style: italic;
        }}

        .markdown-content code {{
            background: var(--bg-tertiary);
            padding: 4px 10px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            font-size: 0.9em;
            color: var(--accent-purple);
            font-weight: 500;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
        }}

        .markdown-content pre {{
            background: #1F2937;
            color: #F9FAFB;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
            margin: 20px 0;
            box-shadow: var(--shadow-md);
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .markdown-content pre code {{
            background: transparent;
            padding: 0;
            color: inherit;
            font-size: 0.95em;
        }}

        .markdown-content blockquote {{
            border-left: 4px solid var(--accent-purple);
            padding: 20px 25px;
            margin: 25px 0;
            color: var(--text-secondary);
            font-style: italic;
            background: linear-gradient(90deg, rgba(124, 58, 237, 0.05), transparent);
            border-radius: 0 8px 8px 0;
        }}

        .markdown-content a {{
            color: var(--accent-purple);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s ease;
            position: relative;
        }}

        .markdown-content a:hover {{
            color: var(--accent-violet);
            text-decoration: underline;
        }}

        .markdown-content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: var(--shadow-sm);
            border-radius: 8px;
            overflow: hidden;
        }}

        .markdown-content th {{
            background: var(--primary-gradient);
            color: white;
            padding: 15px;
            font-weight: 600;
            text-align: left;
        }}

        .markdown-content td {{
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-primary);
        }}

        .markdown-content tr:last-child td {{
            border-bottom: none;
        }}

        .markdown-content tr:nth-child(even) td {{
            background: var(--bg-secondary);
        }}

        .markdown-content tr:hover td {{
            background: var(--bg-tertiary);
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}

            .container {{
                margin: 10px auto;
                border-radius: 16px;
            }}

            .header {{
                padding: 40px 20px;
            }}

            .header h1 {{
                font-size: 2.2em;
            }}

            .header .meta {{
                font-size: 0.9em;
            }}

            .content {{
                padding: 30px 20px;
            }}

            .section {{
                padding: 25px;
                margin-bottom: 30px;
            }}

            .section h2 {{
                font-size: 1.6em;
            }}

            .section h3 {{
                font-size: 1.3em;
            }}

            .markdown-content h1 {{
                font-size: 1.8em;
            }}

            .markdown-content h2 {{
                font-size: 1.5em;
            }}

            .markdown-content h3 {{
                font-size: 1.3em;
            }}

            .keywords {{
                gap: 8px;
            }}

            .keyword-tag {{
                padding: 8px 14px;
                font-size: 0.85em;
            }}
        }}

        @media (max-width: 480px) {{
            .header h1 {{
                font-size: 1.8em;
            }}

            .section {{
                padding: 20px;
            }}

            .section h2 {{
                font-size: 1.4em;
            }}

            .markdown-content h1 {{
                font-size: 1.5em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Research Report</h1>
            <div class="meta">
                <p><strong>Query:</strong> {original_query}</p>
                <p><strong>Generated:</strong> {current_time}</p>
            </div>
        </div>
        <div class="content" id="content">
            <p style="text-align: center; color: #999;">Loading report...</p>
        </div>
        <div class="footer">
            <p>Generated by Deep Research Agent | Powered by AI</p>
        </div>
    </div>

    <script>
        const contentData = {json.dumps(content_data)};

        function renderMarkdown() {{
            const contentEl = document.getElementById('content');
            contentEl.innerHTML = '';

            if (contentData.background) {{
                const section = document.createElement('div');
                section.className = 'section';
                section.innerHTML = `
                    <h2>🔍 Background Research</h2>
                    <div class="markdown-content">${{marked.parse(contentData.background)}}</div>
                `;
                contentEl.appendChild(section);
            }}

            const sortedIterations = Array.from(new Set([
                ...Object.keys(contentData.keywords || {{}}),
                ...Object.keys(contentData.summaries || {{}}),
                ...Object.keys(contentData.gaps || {{}})
            ])).sort((a, b) => parseInt(a) - parseInt(b));

            sortedIterations.forEach(iteration => {{
                if (contentData.keywords && contentData.keywords[iteration]) {{
                    const section = document.createElement('div');
                    section.className = 'section';
                    const keywords = contentData.keywords[iteration].keywords.map(kw =>
                        `<span class="keyword-tag">${{kw}}</span>`
                    ).join('');
                    section.innerHTML = `
                        <span class="iteration-badge">Iteration ${{iteration}}</span>
                        <h2>🎯 Generated Keywords</h2>
                        <div class="keywords">${{keywords}}</div>
                    `;
                    contentEl.appendChild(section);
                }}
            }});

            sortedIterations.forEach(iteration => {{
                if (contentData.summaries && contentData.summaries[iteration]) {{
                    const section = document.createElement('div');
                    section.className = 'section';
                    const summaries = contentData.summaries[iteration].map(item =>
                        `<h3>Keyword: ${{item.keyword}}</h3>
                         <div class="markdown-content">${{marked.parse(item.summary)}}</div>`
                    ).join('');
                    section.innerHTML = `
                        <span class="iteration-badge">Iteration ${{iteration}}</span>
                        <h2>📋 Search Summaries</h2>
                        ${{summaries}}
                    `;
                    contentEl.appendChild(section);
                }}
            }});

            sortedIterations.forEach(iteration => {{
                if (contentData.gaps && contentData.gaps[iteration]) {{
                    const gap = contentData.gaps[iteration];
                    const status = gap.gaps_found ? '⚠️ Gaps detected' : '✅ No gaps found';
                    const section = document.createElement('div');
                    section.className = 'section';
                    section.innerHTML = `
                        <span class="iteration-badge">Iteration ${{iteration}}</span>
                        <h2>🔎 Gap Analysis</h2>
                        <p><strong>Status:</strong> ${{status}}</p>
                        <div class="markdown-content">${{marked.parse(gap.raw_output)}}</div>
                    `;
                    contentEl.appendChild(section);
                }}
            }});

            if (contentData.final) {{
                const section = document.createElement('div');
                section.className = 'section final-report';
                section.innerHTML = `
                    <h2>📝 Final Report</h2>
                    <div class="markdown-content">${{marked.parse(contentData.final)}}</div>
                `;
                contentEl.appendChild(section);
            }}

            document.querySelectorAll('pre code').forEach((block) => {{
                hljs.highlightElement(block);
            }});
        }}

        document.addEventListener('DOMContentLoaded', renderMarkdown);
    </script>
</body>
</html>"""

    logger.info("HTML report generated successfully")
    return {"html_report": html_content}


# Build the graph
builder = StateGraph(ResearchState)
builder.add_node("background_search", background_search_node)
builder.add_node("generate_keywords", generate_keywords_node)
builder.add_node("multi_search", multi_search_node)
builder.add_node("check_gaps", check_gaps_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("generate_html", generate_html_node)

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
builder.add_edge("synthesize", "generate_html")
builder.add_edge("generate_html", END)

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
        "llm_outputs": {},
        "html_report": "",
    }

    logger.debug("Initial state prepared, starting graph execution")
    result = graph.invoke(initial_state)

    logger.info("=" * 50)
    logger.info(f"RESEARCH AGENT COMPLETED")
    logger.info(f"Final report length: {len(result['final_report'])} characters")
    logger.info("=" * 50)

    return result


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
        default="html",
        choices=["markdown", "txt", "json", "html"],
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
                        "report": report["final_report"],
                        "llm_outputs": report["llm_outputs"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            elif args.format == "html":
                f.write(report["html_report"])
            else:
                f.write(report["final_report"])

        logger.info(f"✅ Report saved to: {output_file}")
    except IOError as e:
        logger.error(f"❌ Failed to write output file: {e}")
        raise

    # Also print to console for immediate feedback
    if args.format == "html":
        print("HTML report generated. Open the file in a browser to view it.")
    else:
        print(report["final_report"])
    logger.info("Research agent execution finished")
