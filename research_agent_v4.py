import warnings

warnings.filterwarnings(
    "ignore",
    message="Field name.*shadows an attribute in parent.*",
    category=UserWarning,
)

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated, Dict, Any, Optional
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


def initialize_research_system():
    """Initialize research agent system (LLM instances and graph)."""
    from config import load_model_config
    from llm_factory import create_llm_instances
    from research_agent_v4 import (
        clarify_query_node,
        background_search_node,
        generate_keywords_node,
        multi_search_node,
        check_gaps_node,
        synthesize_node,
        generate_html_node,
        collect_user_response_node,
        route_after_clarify,
        route_after_check,
    )
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(ResearchState)
    builder.add_node("clarify_query", clarify_query_node)
    builder.add_node("background_search", background_search_node)
    builder.add_node("generate_keywords", generate_keywords_node)
    builder.add_node("multi_search", multi_search_node)
    builder.add_node("check_gaps", check_gaps_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("generate_html", generate_html_node)
    builder.add_node("collect_user_response", collect_user_response_node)

    builder.add_edge(START, "clarify_query")
    builder.add_conditional_edges("clarify_query", route_after_clarify)
    builder.add_edge("collect_user_response", "background_search")
    builder.add_edge("background_search", "generate_keywords")
    builder.add_edge("generate_keywords", "multi_search")
    builder.add_edge("multi_search", "check_gaps")
    builder.add_conditional_edges("check_gaps", route_after_check)
    builder.add_edge("synthesize", "generate_html")
    builder.add_edge("generate_html", END)

    graph = builder.compile()

    logger.info("Research agent system initialized")


def get_llm_instances():
    """Get LLM instances."""
    from research_agent_v4 import (
        clarify_query_node,
        background_search_node,
        generate_keywords_node,
        multi_search_node,
        check_gaps_node,
        synthesize_node,
        generate_html_node,
        collect_user_response_node,
        route_after_clarify,
        route_after_check,
    )

    global llm_instances_global

    if not llm_instances_global:
        from config import load_model_config
        from llm_factory import create_llm_instances

        model_config = load_model_config()
        llm_instances_global = create_llm_instances(model_config)

        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(ResearchState)
        builder.add_node("clarify_query", clarify_query_node)
        builder.add_node("background_search", background_search_node)
        builder.add_node("generate_keywords", generate_keywords_node)
        builder.add_node("multi_search", multi_search_node)
        builder.add_node("check_gaps", check_gaps_node)
        builder.add_node("synthesize", synthesize_node)
        builder.add_node("generate_html", generate_html_node)
        builder.add_node("collect_user_response", collect_user_response_node)

        builder.add_edge(START, "clarify_query")
        builder.add_conditional_edges("clarify_query", route_after_clarify)
        builder.add_edge("collect_user_response", "background_search")
        builder.add_edge("background_search", "generate_keywords")
        builder.add_edge("generate_keywords", "multi_search")
        builder.add_edge("multi_search", "check_gaps")
        builder.add_conditional_edges("check_gaps", route_after_check)
        builder.add_edge("synthesize", "generate_html")
        builder.add_edge("generate_html", END)

        graph = builder.compile()

        logger.info("Research agent system initialized")

    return llm_instances_global


def get_graph():
    """Get research graph."""
    from research_agent_v4 import (
        clarify_query_node,
        background_search_node,
        generate_keywords_node,
        multi_search_node,
        check_gaps_node,
        synthesize_node,
        generate_html_node,
        collect_user_response_node,
        route_after_clarify,
        route_after_check,
    )

    global llm_instances_global

    if not llm_instances_global:
        from config import load_model_config
        from llm_factory import create_llm_instances

        model_config = load_model_config()
        llm_instances_global = create_llm_instances(model_config)

        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(ResearchState)
        builder.add_node("clarify_query", clarify_query_node)
        builder.add_node("background_search", background_search_node)
        builder.add_node("generate_keywords", generate_keywords_node)
        builder.add_node("multi_search", multi_search_node)
        builder.add_node("check_gaps", check_gaps_node)
        builder.add_node("synthesize", synthesize_node)
        builder.add_node("generate_html", generate_html_node)
        builder.add_node("collect_user_response", collect_user_response_node)

        builder.add_edge(START, "clarify_query")
        builder.add_conditional_edges("clarify_query", route_after_clarify)
        builder.add_edge("collect_user_response", "background_search")
        builder.add_edge("background_search", "generate_keywords")
        builder.add_edge("generate_keywords", "multi_search")
        builder.add_edge("multi_search", "check_gaps")
        builder.add_conditional_edges("check_gaps", route_after_check)
        builder.add_edge("synthesize", "generate_html")
        builder.add_edge("generate_html", END)

        graph = builder.compile()

        logger.info("Research agent system initialized")

    return graph


# Global variables for LLM instances and graph
_llm_instances_global: Dict[str, ChatOpenAI] = {}
_graph_global: Any = None


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
    gap_details: str  # Detailed description of identified gaps
    final_report: str
    current_time: str
    iteration: int
    max_iterations: int
    keyword_history: list
    lang: str
    llm_outputs: dict
    html_report: str
    char_limits: dict
    background_sources: list
    keyword_search_sources: dict
    clarification_needed: bool
    user_context: str
    clarification_questions: list
    user_provided_context: str
    user_answers: Optional[dict]
    allow_console_input: bool


# Define tools
@tool
def web_search(query: str) -> str:
    """Search the web using Tavily to find relevant information."""
    search = TavilySearch(max_results=5)
    results = search.invoke({"query": query})

    if isinstance(results, dict):
        return json.dumps(results, ensure_ascii=False)
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

    extracted_sources = []
    try:
        results_dict = (
            json.loads(search_results)
            if isinstance(search_results, str)
            else search_results
        )
        if isinstance(results_dict, dict) and "results" in results_dict:
            for result in results_dict["results"]:
                if "url" in result:
                    extracted_sources.append(
                        {
                            "url": result["url"],
                            "title": result.get("title", ""),
                        }
                    )
        logger.info(
            f"Extracted {len(extracted_sources)} sources from background search"
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse search results as JSON: {e}")
    except Exception as e:
        logger.warning(f"Error extracting sources: {e}")

    time_instruction = get_time_awareness_instruction(
        state["current_time"], state["lang"]
    )

    char_limit = state.get("char_limits", {}).get("background", 300)

    sources_text = (
        "\n\nSources:\n"
        + "\n".join(
            [f"- [{source['title']}]({source['url']})" for source in extracted_sources]
        )
        if extracted_sources
        else ""
    )

    summary_instructions = {
        "en": f"""{time_instruction}
🎯 USER'S ORIGINAL REQUEST: {state["original_query"]}
💡 IMPORTANT: The background summary should provide essential context and foundational knowledge about the topic.

SUMMARY GUIDELINES:
- Target length: approximately {char_limit} characters
- Use narrative paragraphs, NOT tables
- Avoid markdown table syntax (| symbol, --- separators)
- Use bullet points or numbered lists for structured information
- Focus on key concepts, trends, and relevant context
- Include only essential information to answer the original query

CONTENT FORMAT:
- Start with a brief overview paragraph
- Use bullet points for key facts or trends
- End with any notable observations or gaps
- Do NOT use tables - convert table-like information to bulleted lists

Summarize the following search results for the query '{{query}}' into a comprehensive background summary:
{{search_results}}
{sources_text}""",
        "zh": f"""{time_instruction}
🎯 用户原始请求: {state["original_query"]}
💡 重要: 背景摘要应该提供关于主题的基本上下文和基础知识。

摘要指南:
- 目标长度: 大约 {char_limit} 个字符
- 使用叙述段落，而不是表格
- 避免markdown表格语法（|符号，---分隔符）
- 使用项目符号或编号列表来组织结构化信息
- 专注于关键概念、趋势和相关上下文
- 仅包含回答原始查询所需的基本信息

内容格式:
- 以简短的概述段落开始
- 使用项目符号列出关键事实或趋势
- 以任何值得注意的观察或差距结束
- 不要使用表格 - 将类似表格的信息转换为项目符号列表

将以下搜索结果汇总为查询'{{query}}'的全面背景摘要:
{{search_results}}
{sources_text}""",
    }

    lang = state["lang"]

    prompt = ChatPromptTemplate.from_template(summary_instructions[lang])

    chain = prompt | llm_instances["background_search"] | StrOutputParser()

    logger.debug("Invoking LLM to generate background summary")
    background = chain.invoke({"query": query, "search_results": search_results})
    logger.info(f"Background summary generated: {len(background)} characters")
    logger.debug(f"Background summary output: {background}")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs["background_search"] = {
        "summary": background,
        "sources": extracted_sources,
    }

    return {
        "background": background,
        "confirmed": True,
        "llm_outputs": llm_outputs,
        "background_sources": extracted_sources,
    }


def generate_keywords_node(state: ResearchState):
    logger.info("=== GENERATE KEYWORDS NODE STARTED ===")
    background = state["background"]
    query = state["query"]
    original_query = state["original_query"]
    iteration = state["iteration"]
    keyword_history = state["keyword_history"]
    max_iterations = state["max_iterations"]
    gap_details = state.get("gap_details", "")

    logger.info(
        f"Generating keywords for query: {query} (Iteration {iteration}/{max_iterations})"
    )
    logger.debug(f"Original query: {original_query}")
    logger.debug(f"Background context length: {len(str(background))} characters")
    logger.debug(f"Keyword history: {keyword_history}")
    if gap_details:
        logger.info(f"Previous gap analysis identified gaps: {gap_details[:150]}...")

    time_instruction = get_time_awareness_instruction(
        state["current_time"], state["lang"]
    )

    lang = state["lang"]

    # Build gap-focused instruction if we have gap details from previous iteration
    gap_instruction = ""
    if gap_details and iteration > 1:
        # Escape curly braces in gap_details to prevent LangChain template variable interpretation
        escaped_gap_details = gap_details.replace("{", "{{").replace("}", "}}")
        gap_instruction_en = f"""

🔍 GAP ANALYSIS FROM PREVIOUS ITERATION:
The previous research identified the following gaps that need to be filled:
{escaped_gap_details}

⚠️ CRITICAL: Generate keywords SPECIFICALLY to address these gaps. Your keywords should target the missing information described above."""
        gap_instruction_zh = f"""

🔍 前一次迭代的缺口分析：
前一次研究识别出以下需要填补的缺口：
{escaped_gap_details}

⚠️ 关键：生成专门针对这些缺口的关键词。你的关键词应该针对上述描述的缺失信息。"""
        gap_instruction = gap_instruction_en if lang == "en" else gap_instruction_zh

    base_instruction = (
        time_instruction
        + "\n🎯 USER'S ORIGINAL REQUEST: "
        + original_query
        + gap_instruction
    )
    keyword_generation_instructions = {
        "en": base_instruction
        + """

💡 IMPORTANT: Your search should help answer their specific question, not just collect general information about the topic.

📋 INSTRUCTIONS:
1. Generate 3-5 search keywords for Google search in-depth research on: {{query}}
2. Based on background information: {{background}}
3. Keywords must be different from previous searches: """
        + str(keyword_history)
        + """
4. CRITICAL: Generate keywords in MULTIPLE LANGUAGES for better coverage:
    - Include English keywords (primary)
    - Include Chinese keywords if relevant to the query
    - Include other language keywords if helpful for comprehensive research
5. Keywords should be context-aware and specific to user's actual needs
6. Avoid generic/broad terms; focus on specific, relevant terms"""
        + (
            "\n7. PRIORITY: Focus on keywords that will fill the identified gaps mentioned above."
            if gap_details and iteration > 1
            else ""
        )
        + """

OUTPUT FORMAT (JSON only):
Return a JSON object with a "keywords" key containing an array of 3-5 keyword strings. Do not use markdown code blocks, explanations, or numbered lists.

Example format:
["keyword1_en", "keyword2_zh", "keyword3_en"]

Examples:
- Query: "AI safety" → ["AI safety research", "人工智能安全", "AI alignment"]
- Query: "machine learning applications" → ["machine learning applications", "机器学习应用", "machine learning use cases"]
- Query: "blockchain technology" → ["blockchain technology", "区块链技术", "distributed ledger technology"]

Do NOT include markdown code blocks, explanations, or numbered lists.
""",
        "zh": base_instruction
        + """

💡 重要: 你的搜索应该帮助回答他们的具体问题，而不仅仅是收集关于主题的一般信息。

📋 指令:
1. 为以下查询生成3-5个搜索关键词: {{query}}
2. 基于背景信息: {{background}}
3. 关键词必须与之前的搜索不同: """
        + str(keyword_history)
        + """
4. 重要: 生成多语言关键词以获得更好的覆盖:
    - 包含英文关键词（主要）
    - 包含中文关键词（如果与查询相关）
    - 包含其他语言关键词（如果有助于全面研究）
5. 关键词应该对应用户实际需求的上下文相关
6. 避免通用/广泛的术语；专注于具体、相关的词汇"""
        + (
            "\n7. 优先级：专注于能够填补上述识别出的缺口的关键词。"
            if gap_details and iteration > 1
            else ""
        )
        + """

输出格式（仅JSON）：
返回JSON对象，必须包含一个"keywords"键，其值是一个包含3-5个关键词的字符串数组。不要使用markdown代码块、解释或编号列表。

示例格式：
["keyword1_en", "keyword2_zh", "keyword3_en"]

示例:
- 查询: "AI安全" → ["AI safety research", "人工智能安全", "AI alignment"]
- 查询: "machine learning applications" → ["machine learning applications", "机器学习应用", "machine learning use cases"]
- 查询: "区块链技术" → ["blockchain technology", "区块链技术", "distributed ledger technology"]
""",
    }

    prompt = ChatPromptTemplate.from_template(keyword_generation_instructions[lang])

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

    char_limit = state.get("char_limits", {}).get("keyword_summary", 500)

    multi_search_outputs = []
    keyword_search_sources = state.get("keyword_search_sources", {})

    for idx, kw in enumerate(keywords, 1):
        logger.info(f"Processing keyword {idx}/{len(keywords)}: {kw}")

        search_tool = web_search
        logger.debug(f"Searching for keyword: {kw}")
        result = search_tool.invoke(kw)
        logger.debug(f"Search results for '{kw}': {len(str(result))} characters")

        extracted_sources = []
        try:
            results_dict = json.loads(result) if isinstance(result, str) else result
            if isinstance(results_dict, dict) and "results" in results_dict:
                for res in results_dict["results"]:
                    if "url" in res:
                        extracted_sources.append(
                            {
                                "url": res["url"],
                                "title": res.get("title", ""),
                            }
                        )
            logger.debug(
                f"Extracted {len(extracted_sources)} sources for keyword '{kw}'"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse search results for '{kw}': {e}")
        except Exception as e:
            logger.warning(f"Error extracting sources for '{kw}': {e}")

        sources_list = (
            "\n".join(
                [
                    f"- [{source['title']}]({source['url']})"
                    for source in extracted_sources
                ]
            )
            if extracted_sources
            else ""
        )

        sources_text = f"\n\nSources:\n{sources_list}" if sources_list else ""

        summary_format_instructions = {
            "en": f"""{get_time_awareness_instruction(state["current_time"], state["lang"])}
🎯 USER'S ORIGINAL REQUEST: {original_query}
💡 IMPORTANT: The paragraph must help answer the user's original research question. Focus on information relevant to their specific inquiry.

SUMMARY GUIDELINES:
- Target length: approximately {char_limit} characters
- Use narrative paragraphs, NOT tables
- Avoid markdown table syntax (| symbol, --- separators)
- Use bullet points or numbered lists for structured information
- Focus on key findings directly relevant to keyword
- Highlight insights that help answer original query

CONTENT FORMAT:
- Start with a concise summary paragraph
- Use bullet points for specific details or findings
- Do NOT use tables - convert table-like information to bulleted lists
- End with any key takeaways relevant to original query

Summarize the following search results for keyword '{{kw}}': {{result}}{sources_text}""",
            "zh": f"""{get_time_awareness_instruction(state["current_time"], state["lang"])}
🎯 用户原始请求: {original_query}
💡 重要: 段落必须帮助回答用户的原始研究问题。专注于与他们的具体询问相关的信息。

摘要指南:
- 目标长度: 大约 {char_limit} 个字符
- 使用叙述段落，而不是表格
- 避免markdown表格语法（|符号，---分隔符）
- 使用项目符号或编号列表来组织结构化信息
- 专注于与关键词直接相关的关键发现
- 突出有助于回答原始查询的见解

内容格式:
- 以简明摘要段落开始
- 使用项目符号列出具体细节或发现
- 不要使用表格 - 将类似表格的信息转换为项目符号列表
- 以与原始查询相关的任何关键要点结束

总结以下搜索结果的关键词'{{kw}}': {{result}}{sources_text}""",
        }

        lang = state["lang"]

        prompt = ChatPromptTemplate.from_template(summary_format_instructions[lang])
        chain = prompt | llm_instances["multi_search"] | StrOutputParser()

        logger.debug(f"Summarizing results for keyword: {kw}")
        summary = chain.invoke({"kw": kw, "result": result})
        logger.info(f"Generated summary for '{kw}': {len(summary)} characters")
        logger.debug(f"Summary output for '{kw}': {summary}")
        summaries.append(summary)
        multi_search_outputs.append(
            {
                "keyword": kw,
                "summary": summary,
                "sources": extracted_sources,
            }
        )

        keyword_search_sources[kw] = extracted_sources

    logger.info(f"Multi-search completed. Generated {len(summaries)} summaries")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs[f"multi_search_iteration_{state['iteration'] - 1}"] = (
        multi_search_outputs
    )

    return {
        "summaries": summaries,
        "llm_outputs": llm_outputs,
        "keyword_search_sources": keyword_search_sources,
    }


def check_gaps_node(state: ResearchState):
    logger.info("=== CHECK GAPS NODE STARTED ===")
    summaries = state["summaries"]
    query = state["query"]
    original_query = state["original_query"]
    lang = state["lang"]

    logger.info(f"Checking gaps for {len(summaries)} summaries against query: {query}")
    logger.debug(f"Original query context: {original_query}")

    time_instruction = get_time_awareness_instruction(state["current_time"], lang)

    gap_analysis_instructions = {
        "en": f"""{time_instruction}
🎯 USER'S ORIGINAL REQUEST: {original_query}

💡 CRITICAL: Analyze whether the current research summaries sufficiently address the user's original question. Identify specific deficiencies and gaps in the research.

Review these summaries:
{{summaries}}

TASK:
1. Determine if there are gaps in the research (yes/no)
2. If gaps exist, identify SPECIFICALLY what information is missing
3. Describe what additional research would be needed to fill these gaps

OUTPUT FORMAT (JSON only):
{{{{"gaps_found": true/false, "gap_details": "Detailed description of what gaps were identified and what specific information is missing. If no gaps, leave empty."}}}}

Examples of gap analysis:
- Query: "AI safety developments 2024" 
  → Gap: "Missing specific 2024 regulatory updates and recent safety incidents"
- Query: "Machine learning in healthcare"
  → Gap: "No information about clinical trial results or FDA approvals"
- Query: "Climate change policy"
  → Gap: "Lacks comparison between EU and US policy approaches"

Respond with JSON only, no markdown code blocks.""",
        "zh": f"""{time_instruction}
🎯 用户原始请求: {original_query}

💡 关键：分析当前研究摘要是否充分回答了用户的原始问题。识别研究中存在的具体缺陷和缺口。

查看这些摘要：
{{summaries}}

任务：
1. 确定研究中是否存在缺口（是/否）
2. 如果存在缺口，具体指出缺少哪些信息
3. 描述需要哪些额外研究来填补这些缺口

输出格式（仅JSON）：
{{{{"gaps_found": true/false, "gap_details": "对识别出的缺口和缺少的具体信息的详细描述。如果没有缺口，留空。"}}}}

缺口分析示例：
- 查询: "2024年AI安全发展"
  → 缺口: "缺少2024年具体监管更新和近期安全事件的信息"
- 查询: "医疗保健中的机器学习"
  → 缺口: "没有关于临床试验结果或FDA批准的信息"
- 查询: "气候变化政策"
  → 缺口: "缺乏欧盟和美国政策方法的比较"

仅返回JSON，不要markdown代码块。""",
    }

    prompt = ChatPromptTemplate.from_template(gap_analysis_instructions[lang])
    chain = prompt | llm_instances["check_gaps"] | StrOutputParser()

    logger.debug("Invoking LLM to check for gaps")
    llm_output = chain.invoke({"summaries": "\n".join(summaries), "query": query})
    logger.debug(f"Gap analysis raw output: {llm_output}")

    gaps_found = False
    gap_details = ""

    try:
        json_match = re.search(r"\{{[\s\S]*?\}}", llm_output)
        if json_match:
            json_str = json_match.group(0)
            logger.debug(f"Extracted JSON: {json_str}")
            parsed = json.loads(json_str)
            gaps_found = parsed.get("gaps_found", False)
            gap_details = parsed.get("gap_details", "")
        else:
            gaps_found = "yes" in llm_output.lower() or "true" in llm_output.lower()
            gap_details = llm_output if gaps_found else ""
            logger.warning("No JSON found in gap analysis, using fallback parsing")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error in gap analysis: {e}, using fallback")
        gaps_found = "yes" in llm_output.lower() or "true" in llm_output.lower()
        gap_details = llm_output if gaps_found else ""
    except Exception as e:
        logger.warning(f"Error parsing gap analysis: {e}")
        gaps_found = "yes" in llm_output.lower() or "true" in llm_output.lower()
        gap_details = llm_output if gaps_found else ""

    logger.info(f"Gap analysis complete. Gaps found: {gaps_found}")
    if gap_details:
        logger.info(f"Gap details: {gap_details[:200]}...")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs[f"check_gaps_iteration_{state['iteration'] - 1}"] = {
        "raw_output": llm_output,
        "gaps_found": gaps_found,
        "gap_details": gap_details,
    }

    return {
        "gaps_found": gaps_found,
        "gap_details": gap_details,
        "llm_outputs": llm_outputs,
    }


def clarify_query_node(state: ResearchState):
    logger.info("=== CLARIFY QUERY NODE STARTED ===")
    query = state["query"]
    original_query = state["original_query"]
    lang = state["lang"]

    logger.info(f"Analyzing query for clarification: {query}")
    logger.debug(f"Original query: {original_query}")
    logger.debug(f"Language: {lang}")

    time_instruction = get_time_awareness_instruction(
        state["current_time"], state["lang"]
    )

    clarification_instructions = {
        "en": time_instruction
        + """🎯 USER'S QUERY: {original_query}

Analyze the user's query and determine if it needs clarification.

YOUR TASK:
1. Evaluate if the query is clear enough to conduct effective research
2. Identify missing context, background information, or ambiguity
3. Generate specific clarification questions that would improve research quality

CLARIFICATION CRITERIA:
- Needs clarification if query is vague, ambiguous, or lacks specific context
- Needs clarification if topic is broad and could be interpreted multiple ways
- Needs clarification if time/scope/geography/context is unclear
- Does NOT need clarification if query is specific and well-defined

OUTPUT FORMAT (JSON only):
Return a JSON object with keys: clarification_needed (boolean), reasoning (string), questions (array).

Examples where clarification is NEEDED:
- "Machine learning" → Too broad, needs context on specific area/application
- "Climate change" → Could refer to science, policy, impacts, etc.
- "Latest developments in AI" → What specific aspect? When is latest?

Examples where clarification is NOT needed:
- "Impact of GPT-4 on software development"
- "Quantum computing applications in healthcare 2024"
- "Comparison of Python vs JavaScript for web development in 2025"
""",
    }

    prompt = ChatPromptTemplate.from_template(clarification_instructions[lang])
    chain = prompt | llm_instances["clarify_query"] | StrOutputParser()

    logger.debug("Invoking LLM to analyze query for clarification")
    llm_output = chain.invoke({"original_query": original_query})
    logger.debug(f"Clarification analysis result: {llm_output}")

    clarification_needed = False
    reasoning = ""
    questions = []

    try:
        json_match = re.search(r"\{[\s\S]*?\}", llm_output)
        if json_match:
            json_str = json_match.group(0)
            logger.debug(f"Extracted JSON: {json_str}")
            parsed = json.loads(json_str)
            clarification_needed = parsed.get("clarification_needed", False)
            reasoning = parsed.get("reasoning", "")
            questions = parsed.get("questions", [])
        else:
            logger.warning("No JSON found, attempting direct text analysis")
            if "clarification_needed" in llm_output.lower():
                clarification_needed = "true" in llm_output.lower()
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
    except Exception as e:
        logger.warning(f"Error parsing clarification result: {e}")

    logger.info(f"Clarification needed: {clarification_needed}")
    logger.info(f"Reasoning: {reasoning}")
    logger.info(f"Generated {len(questions)} clarification questions: {questions}")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs["clarify_query"] = {
        "raw_output": llm_output,
        "clarification_needed": clarification_needed,
        "reasoning": reasoning,
        "questions": questions,
    }

    return {
        "clarification_needed": clarification_needed,
        "user_context": reasoning,
        "clarification_questions": questions,
        "llm_outputs": llm_outputs,
    }


def collect_user_response_node(state: ResearchState):
    logger.info("=== COLLECT USER RESPONSE NODE STARTED ===")
    questions = state["clarification_questions"]
    original_query = state["original_query"]
    pre_provided_answers = state.get("user_answers")

    if not questions:
        logger.info(
            "No clarification questions to ask. Proceeding with original query."
        )
        return {"user_provided_context": ""}

    if pre_provided_answers:
        logger.info("Using pre-provided user answers from state")
        user_answers_list = []
        for idx, question in enumerate(questions, 1):
            answer = pre_provided_answers.get(str(idx), "")
            if answer:
                user_answers_list.append(f"Q{idx}. {question}\nA{idx}. {answer}")

        if user_answers_list:
            user_provided_context = "\n\n".join(user_answers_list)
            logger.info(
                f"Collected {len(user_answers_list)} clarification answers from pre-provided input"
            )

            clarifications_text = " ".join(
                [f"({answer})" for answer in pre_provided_answers.values() if answer]
            )
            enhanced_query = f"{original_query} {clarifications_text}"
            logger.info(f"Enhanced query: {enhanced_query}")

            llm_outputs = state.get("llm_outputs", {})
            llm_outputs["collect_user_response"] = {
                "answers": user_answers_list,
                "enhanced_query": enhanced_query,
            }

            return {
                "user_provided_context": user_provided_context,
                "query": enhanced_query,
                "llm_outputs": llm_outputs,
            }
        else:
            logger.info(
                "No answers provided in pre-provided input. Proceeding with original query."
            )
            return {"user_provided_context": ""}

    logger.info("Waiting for user to provide clarification answers...")

    print("\n" + "=" * 60)
    print("CLARIFICATION NEEDED")
    print("=" * 60)
    print("\nTo conduct more accurate and targeted research, please clarify:\n")

    user_answers = []
    for idx, question in enumerate(questions, 1):
        print(f"{idx}. {question}")
        answer = input("   Your answer (press Enter to skip): ").strip()
        if answer:
            user_answers.append(f"Q{idx}. {question}\nA{idx}. {answer}")

    print("\n" + "=" * 60)
    print("Proceeding with research...")
    print("=" * 60 + "\n")

    if user_answers:
        user_provided_context = "\n\n".join(user_answers)
        logger.info(f"Collected {len(user_answers)} clarification answers")

        clarifications_text = " ".join(
            [f"({ans.split('A')[1].strip()})" for ans in user_answers]
        )
        enhanced_query = f"{original_query} {clarifications_text}"
        logger.info(f"Enhanced query: {enhanced_query}")

        llm_outputs = state.get("llm_outputs", {})
        llm_outputs["collect_user_response"] = {
            "answers": user_answers,
            "enhanced_query": enhanced_query,
        }

        return {
            "user_provided_context": user_provided_context,
            "query": enhanced_query,
            "llm_outputs": llm_outputs,
        }
    else:
        logger.info("No answers provided by user. Proceeding with original query.")
        return {"user_provided_context": ""}


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

    char_limit = state.get("char_limits", {}).get("final_report", 2000)

    all_sources = []
    background_sources = state.get("background_sources", [])
    all_sources.extend(background_sources)

    keyword_search_sources = state.get("keyword_search_sources", {})
    for keyword, sources in keyword_search_sources.items():
        all_sources.extend(sources)

    unique_sources = {}
    for source in all_sources:
        url = source["url"]
        if url not in unique_sources:
            unique_sources[url] = source

    sources_text = (
        "\n\n## Sources\n\n"
        + "\n".join(
            [
                f"- [{source['title']}]({source['url']})"
                for source in unique_sources.values()
            ]
        )
        if unique_sources
        else ""
    )

    prompt = ChatPromptTemplate.from_template(
        f"{get_time_awareness_instruction(state['current_time'], state['lang'])}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: Ensure final report comprehensively answers user's original research question. Conclude by summarizing how report addresses their specific inquiry.\n💡 CRITICAL: You MUST include a 'Sources' section at the end of your report with all the reference URLs provided below. This section should list all sources used in your research.\n\\Summarize length: approximately {char_limit} characters.\n\\Synthesize these summaries into a comprehensive report for the query: {{query}}\n\\Summaries: {{summaries}}{sources_text}"
    )
    chain = prompt | llm_instances["synthesize"] | StrOutputParser()

    logger.debug("Invoking LLM to synthesize final report")
    report = chain.invoke({"query": query, "summaries": "\n".join(summaries)})
    logger.info(f"Final report generated: {len(report)} characters")
    logger.debug(f"Final report output: {report}")

    llm_outputs = state.get("llm_outputs", {})
    llm_outputs["synthesize"] = {
        "report": report,
        "sources": list(unique_sources.values()),
    }

    return {"final_report": report, "llm_outputs": llm_outputs}


def save_node_markdown_files(state: dict, output_dir: Path):
    logger.info("=== SAVING NODE MARKDOWN FILES STARTED ===")

    original_query = state.get("original_query", "")
    llm_outputs = state.get("llm_outputs", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    if "background_search" in llm_outputs:
        bg_data = llm_outputs["background_search"]
        bg_summary = (
            bg_data.get("summary", "") if isinstance(bg_data, dict) else str(bg_data)
        )

        sources_md = ""
        if isinstance(bg_data, dict) and "sources" in bg_data:
            sources_md = "\n\n**Sources:**\n"
            for source in bg_data["sources"]:
                sources_md += (
                    f"- [{source.get('title', source['url'])}]({source['url']})\n"
                )

        bg_file = output_dir / "01_background_search.md"
        bg_file.write_text(
            f"# 🔍 Background Research\n\n{bg_summary}{sources_md}",
            encoding="utf-8",
        )
        logger.info(f"Saved background search to: {bg_file}")

    if "synthesize" in llm_outputs:
        synthesize_data = llm_outputs["synthesize"]
        final_report = (
            synthesize_data.get("report", "")
            if isinstance(synthesize_data, dict)
            else str(synthesize_data)
        )

        sources_md = ""
        if isinstance(synthesize_data, dict) and "sources" in synthesize_data:
            sources_md = "\n\n## Sources\n\n"
            for source in synthesize_data["sources"]:
                sources_md += (
                    f"- [{source.get('title', source['url'])}]({source['url']})\n"
                )

        final_file = output_dir / "05_synthesize.md"
        final_file.write_text(
            f"# 📝 Final Report\n\n{final_report}{sources_md}",
            encoding="utf-8",
        )
        logger.info(f"Saved final report to: {final_file}")

    iterations = set()
    for key in llm_outputs.keys():
        if key.startswith("generate_keywords_iteration_"):
            iteration = key.split("_")[-1]
            iterations.add(iteration)

    for iteration in sorted(iterations, key=int):
        keywords_key = f"generate_keywords_iteration_{iteration}"
        multi_search_key = f"multi_search_iteration_{iteration}"
        check_gaps_key = f"check_gaps_iteration_{iteration}"

        if keywords_key in llm_outputs:
            kw_data = llm_outputs[keywords_key]
            keywords = kw_data.get("parsed_keywords", [])
            keywords_file = output_dir / f"02_generate_keywords_iter{iteration}.md"
            keywords_file.write_text(
                f"# 🎯 Generated Keywords (Iteration {iteration})\n\n"
                f"**Original Query:** {original_query}\n\n"
                f"## Keywords\n\n" + "\n".join(f"- {kw}" for kw in keywords),
                encoding="utf-8",
            )
            logger.info(f"Saved keywords iteration {iteration} to: {keywords_file}")

        if multi_search_key in llm_outputs:
            summaries_data = llm_outputs[multi_search_key]
            summaries_md = f"# 📋 Search Summaries (Iteration {iteration})\n\n**Original Query:** {original_query}\n\n"

            for item in summaries_data:
                keyword = item.get("keyword", "")
                summary = item.get("summary", "")
                summaries_md += f"\n## Keyword: {keyword}\n\n{summary}\n\n"

            summaries_file = output_dir / f"03_multi_search_iter{iteration}.md"
            summaries_file.write_text(summaries_md, encoding="utf-8")
            logger.info(f"Saved summaries iteration {iteration} to: {summaries_file}")

        if check_gaps_key in llm_outputs:
            gap_data = llm_outputs[check_gaps_key]
            gaps_found = gap_data.get("gaps_found", False)
            raw_output = gap_data.get("raw_output", "")
            status = "⚠️ Gaps detected" if gaps_found else "✅ No gaps found"

            gaps_file = output_dir / f"04_check_gaps_iter{iteration}.md"
            gaps_file.write_text(
                f"# 🔎 Gap Analysis (Iteration {iteration})\n\n"
                f"**Original Query:** {original_query}\n\n"
                f"**Status:** {status}\n\n"
                f"**Analysis:**\n\n{raw_output}",
                encoding="utf-8",
            )
            logger.info(f"Saved gap analysis iteration {iteration} to: {gaps_file}")

    logger.info("=== SAVING NODE MARKDOWN FILES COMPLETED ===")


def generate_html_node(state: ResearchState):
    logger.info("=== GENERATE HTML NODE STARTED ===")

    original_query = state["original_query"]
    current_time = state["current_time"]
    llm_outputs = state.get("llm_outputs", {})

    logger.info("Generating HTML report")

    def escape_js_string(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        return text

    content_data = {}

    if "background_search" in llm_outputs:
        bg_data = llm_outputs["background_search"]
        if isinstance(bg_data, dict):
            content_data["background"] = bg_data.get("summary", "")
        else:
            content_data["background"] = bg_data

    for key in ["background", "final"]:
        if key in content_data:
            content_data[key] = escape_js_string(content_data[key])

    if "summaries" in content_data:
        for iter_key, summaries in content_data["summaries"].items():
            for item in summaries:
                if "summary" in item:
                    item["summary"] = escape_js_string(item["summary"])

    if "gaps" in content_data:
        for iter_key, gap_data in content_data["gaps"].items():
            if "raw_output" in gap_data:
                gap_data["raw_output"] = escape_js_string(gap_data["raw_output"])

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
                    "summary": item["summary"],
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
                "raw_output": value.get("raw_output", ""),
            }

    if "synthesize" in llm_outputs:
        synthesize_data = llm_outputs["synthesize"]
        if isinstance(synthesize_data, dict):
            content_data["final"] = synthesize_data.get("report", "")
            content_data["sources"] = [
                {
                    "url": source.get("url", ""),
                    "title": source.get("title", ""),
                }
                for source in synthesize_data.get("sources", [])
            ]
        else:
            content_data["final"] = synthesize_data
            content_data["sources"] = []

    if "background_search" in llm_outputs:
        bg_data = llm_outputs["background_search"]
        if isinstance(bg_data, dict) and "sources" in bg_data:
            if "sources" not in content_data:
                content_data["sources"] = []
            seen_urls = {s.get("url") for s in content_data["sources"] if "url" in s}
            for source in bg_data["sources"]:
                if "url" in source and source["url"] not in seen_urls:
                    content_data["sources"].append(
                        {
                            "url": source.get("url", ""),
                            "title": source.get("title", ""),
                        }
                    )
                    seen_urls.add(source["url"])

    js_data = json.dumps(
        {
            "background": escape_js_string(content_data.get("background", "")),
            "keywords": content_data.get("keywords", {}),
            "summaries": content_data.get("summaries", {}),
            "gaps": content_data.get("gaps", {}),
            "final": escape_js_string(content_data.get("final", "")),
            "sources": content_data.get("sources", []),
        }
    )

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
            background: var(--bg-primary);
            color: var(--text-primary);
            border-left: none;
            position: relative;
        }}

        .final-report::before {{
            background: radial-gradient(circle, rgba(124, 58, 237, 0.05) 0%, transparent 70%);
        }}

        .final-report h2 {{
            color: var(--accent-purple);
            border-bottom: 3px solid var(--border-color);
            font-family: 'Playfair Display', Georgia, serif;
        }}

        .final-report h2::after {{
            background: var(--secondary-gradient);
        }}

        .final-report .markdown-content {{
            color: var(--text-primary);
        }}

        .final-report .markdown-content strong {{
            color: var(--accent-purple);
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

        .section ul {{
            list-style-type: none;
            padding-left: 0;
        }}

        .section ul li {{
            margin-bottom: 10px;
        }}

        .section ul li a {{
            color: var(--accent-purple);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s ease;
            display: inline-block;
        }}

        .section ul li a:hover {{
            color: var(--accent-violet);
            text-decoration: underline;
        }}

        .section-divider {{
            height: 3px;
            background: linear-gradient(90deg, transparent, var(--primary-gradient), transparent);
            border-radius: 2px;
            margin: 50px 0;
            position: relative;
        }}

        .section-divider::before {{
            content: '';
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 100px;
            height: 3px;
            background: white;
            border-radius: 2px;
            box-shadow: 0 0 10px rgba(124, 58, 237, 0.5);
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

        /* Sticky Navigation Bar Styles */
        .nav-bar {{
            position: sticky;
            top: 0;
            z-index: 1000;
            background: var(--bg-primary);
            border-bottom: 2px solid var(--border-color);
            box-shadow: var(--shadow-md);
            padding: 0;
            transition: all 0.3s ease;
        }}

        .nav-bar.scrolled {{
            box-shadow: var(--shadow-lg);
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(10px);
        }}

        .nav-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 0;
            max-width: 100%;
            overflow-x: auto;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }}

        .nav-container::-webkit-scrollbar {{
            display: none;
        }}

        .nav-links {{
            display: flex;
            list-style: none;
            gap: 0;
            padding: 0;
            margin: 0;
            width: 100%;
            justify-content: space-around;
        }}

        .nav-links li {{
            flex: 1;
            text-align: center;
        }}

        .nav-link {{
            display: block;
            padding: 18px 20px;
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95em;
            transition: all 0.3s ease;
            position: relative;
            white-space: nowrap;
            letter-spacing: 0.3px;
        }}

        .nav-link::before {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 3px;
            background: var(--primary-gradient);
            transition: all 0.3s ease;
            transform: translateX(-50%);
            border-radius: 3px 3px 0 0;
        }}

        .nav-link:hover {{
            color: var(--accent-purple);
            background: linear-gradient(180deg, rgba(124, 58, 237, 0.05) 0%, transparent 100%);
        }}

        .nav-link:hover::before {{
            width: 80%;
        }}

        .nav-link.active {{
            color: var(--accent-purple);
            font-weight: 600;
            background: linear-gradient(180deg, rgba(124, 58, 237, 0.08) 0%, transparent 100%);
        }}

        .nav-link.active::before {{
            width: 80%;
            height: 3px;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.4);
        }}

        .nav-link span {{
            display: inline-block;
            transition: transform 0.3s ease;
        }}

        .nav-link:hover span {{
            transform: translateY(-2px);
        }}

        .mobile-menu-toggle {{
            display: none;
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1001;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: var(--shadow-xl);
            transition: all 0.3s ease;
        }}

        .mobile-menu-toggle:hover {{
            transform: scale(1.1);
            box-shadow: var(--shadow-2xl);
        }}

        .mobile-menu-toggle.active {{
            transform: rotate(45deg);
        }}

        /* Smooth scroll behavior */
        html {{
            scroll-behavior: smooth;
            scroll-padding-top: 80px;
        }}

        /* Section ID targeting for smooth scroll */
        .section {{
            scroll-margin-top: 80px;
        }}

        @media (max-width: 768px) {{
            .nav-container {{
                padding: 0;
            }}

            .nav-link {{
                padding: 15px 12px;
                font-size: 0.85em;
            }}

            .nav-link::before {{
                height: 2px;
            }}

            .nav-link.active::before {{
                height: 2px;
            }}

            .nav-link span {{
                display: block;
                transform: none;
            }}

            .nav-link:hover span {{
                transform: none;
            }}

            html {{
                scroll-padding-top: 60px;
            }}

            .section {{
                scroll-margin-top: 60px;
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

            .nav-container {{
                justify-content: flex-start;
            }}

            .nav-link {{
                padding: 14px 10px;
                font-size: 0.8em;
                letter-spacing: 0;
            }}

            .nav-link span {{
                display: block;
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
        <nav class="nav-bar" id="navBar">
            <div class="nav-container">
                <ul class="nav-links">
                    <li><a href="#background" class="nav-link" data-section="background"><span>🔍 Background</span></a></li>
                    <li><a href="#keywords" class="nav-link" data-section="keywords"><span>🎯 Keywords</span></a></li>
                    <li><a href="#summaries" class="nav-link" data-section="summaries"><span>📋 Summaries</span></a></li>
                    <li><a href="#gaps" class="nav-link" data-section="gaps"><span>🔎 Gap Analysis</span></a></li>
                    <li><a href="#final" class="nav-link" data-section="final"><span>📝 Final Report</span></a></li>
                    <li><a href="#sources" class="nav-link" data-section="sources"><span>📚 Sources</span></a></li>
                </ul>
            </div>
        </nav>
        <div class="content" id="content">
            <p style="text-align: center; color: #999;">Loading report...</p>
        </div>
        <div class="footer">
            <p>Generated by Deep Research Agent | Powered by AI</p>
        </div>
    </div>

    <!-- Store raw markdown content to preserve formatting including code blocks -->
    <script type="text/markdown" id="md-background">{content_data.get("background", "")}</script>
    <script type="text/markdown" id="md-final">{content_data.get("final", "")}</script>
"""

    # Add iteration summaries markdown
    for k, v in content_data.get("summaries", {}).items():
        for idx, item in enumerate(v):
            html_content += f'    <script type="text/markdown" id="md-iteration-{k}-summary-{idx}">{item["summary"]}</script>\n'

    # Add iteration gaps markdown
    for k, v in content_data.get("gaps", {}).items():
        html_content += f'    <script type="text/markdown" id="md-iteration-{k}-gaps">{v["raw_output"]}</script>\n'

    # Build JavaScript data object for HTML
    js_data = json.dumps(
        {
            "background": escape_js_string(content_data.get("background", "")),
            "keywords": content_data.get("keywords", {}),
            "summaries": content_data.get("summaries", {}),
            "gaps": content_data.get("gaps", {}),
            "final": escape_js_string(content_data.get("final", "")),
            "sources": content_data.get("sources", []),
        }
    )

    html_content += f"""
    <script>
        const contentData = {js_data};

        // Configure marked.js for markdown rendering with highlight.js
        marked.setOptions({{
            breaks: true,
            gfm: true,
            highlight: function(code, lang) {{
                if (lang && hljs.getLanguage(lang)) {{
                    try {{
                        return hljs.highlight(code, {{ language: lang }}).value;
                    }} catch (err) {{
                        console.error('Highlight.js error:', err);
                    }}
                }}
                return hljs.highlightAuto(code).value;
            }}
        }});

        function renderMarkdown() {{
            const contentEl = document.getElementById('content');
            contentEl.innerHTML = '';

            function addDivider() {{
                const divider = document.createElement('div');
                divider.className = 'section-divider';
                contentEl.appendChild(divider);
            }}

            if (contentData.background) {{
                const section = document.createElement('div');
                section.className = 'section';
                section.id = 'background';
                section.innerHTML = `
                    <h2>🔍 Background Research</h2>
                    <div class="markdown-content">${{marked.parse(contentData.background)}}</div>
                `;
                contentEl.appendChild(section);
                addDivider();
            }}

            const sortedIterations = Array.from(new Set([
                ...Object.keys(contentData.keywords || {{}}),
                ...Object.keys(contentData.summaries || {{}}),
                ...Object.keys(contentData.gaps || {{}})
            ])).sort((a, b) => parseInt(a) - parseInt(b));

            const lastIteration = sortedIterations.length > 0 ? sortedIterations[sortedIterations.length - 1] : null;

            sortedIterations.forEach(iteration => {{
                if (contentData.keywords && contentData.keywords[iteration]) {{
                    const section = document.createElement('div');
                    section.className = 'section';
                    section.id = 'keywords';
                    const keywords = contentData.keywords[iteration].keywords.map(kw =>
                        `<span class="keyword-tag">${{kw}}</span>`
                    ).join('');
                    section.innerHTML = `
                        <span class="iteration-badge">Iteration ${{iteration}}</span>
                        <h2>🎯 Generated Keywords</h2>
                        <div class="keywords">${{keywords}}</div>
                    `;
                    contentEl.appendChild(section);
                    if (iteration === lastIteration) {{
                        addDivider();
                    }}
                }}
            }});

            sortedIterations.forEach(iteration => {{
                if (contentData.summaries && contentData.summaries[iteration]) {{
                    const section = document.createElement('div');
                    section.className = 'section';
                    section.id = 'summaries';
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
                    if (iteration === lastIteration) {{
                        addDivider();
                    }}
                }}
            }});

            sortedIterations.forEach(iteration => {{
                if (contentData.gaps && contentData.gaps[iteration]) {{
                    const gap = contentData.gaps[iteration];
                    const status = gap.gaps_found ? '⚠️ Gaps detected' : '✅ No gaps found';
                    const section = document.createElement('div');
                    section.className = 'section';
                    section.id = 'gaps';
                    section.innerHTML = `
                        <span class="iteration-badge">Iteration ${{iteration}}</span>
                        <h2>🔎 Gap Analysis</h2>
                        <p><strong>Status:</strong> ${{status}}</p>
                        <div class="markdown-content">${{marked.parse(gap.raw_output)}}</div>
                    `;
                    contentEl.appendChild(section);
                    if (iteration === lastIteration) {{
                        addDivider();
                    }}
                }}
            }});

            if (contentData.final) {{
                const section = document.createElement('div');
                section.className = 'section final-report';
                section.id = 'final';
                section.innerHTML = `
                    <h2>📝 Final Report</h2>
                    <div class="markdown-content">${{marked.parse(contentData.final)}}</div>
                `;
                contentEl.appendChild(section);
                addDivider();
            }}

            if (contentData.sources && contentData.sources.length > 0) {{
                const section = document.createElement('div');
                section.className = 'section';
                section.id = 'sources';
                const sourcesList = contentData.sources.map(source =>
                    `<li><a href="${{source.url}}" target="_blank" rel="noopener noreferrer">${{source.title || source.url}}</a></li>`
                ).join('');
                section.innerHTML = `
                    <h2>📚 Sources</h2>
                    <ul style="list-style-type: none; padding-left: 0;">
                        ${{sourcesList}}
                    </ul>
                `;
                contentEl.appendChild(section);
            }}

            document.querySelectorAll('pre code').forEach((block) => {{
                hljs.highlightElement(block);
            }});

            // Initialize navigation after content is loaded
            initNavigation();
        }}

        // Navigation functionality
        function initNavigation() {{
            const navBar = document.getElementById('navBar');
            const navLinks = document.querySelectorAll('.nav-link');
            const sections = document.querySelectorAll('.section');

            // Smooth scroll handling
            navLinks.forEach(link => {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetSection = document.getElementById(targetId);

                    if (targetSection) {{
                        const navHeight = navBar.offsetHeight;
                        const elementPosition = targetSection.getBoundingClientRect().top;
                        const offsetPosition = elementPosition + window.pageYOffset - navHeight - 10;

                        window.scrollTo({{
                            top: offsetPosition,
                            behavior: 'smooth'
                        }});
                    }}
                }});
            }});

            // Active section detection on scroll
            function updateActiveNav() {{
                const scrollPosition = window.pageYOffset + navBar.offsetHeight + 20;

                let currentSection = '';

                sections.forEach(section => {{
                    const sectionTop = section.offsetTop;
                    const sectionHeight = section.offsetHeight;

                    if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {{
                        currentSection = section.id;
                    }}
                }});

                // If we're at the bottom of the page, activate the last section
                if (window.innerHeight + window.pageYOffset >= document.body.offsetHeight - 100) {{
                    currentSection = sections[sections.length - 1].id;
                }}

                navLinks.forEach(link => {{
                    link.classList.remove('active');
                    if (link.getAttribute('data-section') === currentSection) {{
                        link.classList.add('active');
                    }}
                }});
            }}

            // Sticky nav bar scroll effect
            function updateNavBarStyle() {{
                if (window.pageYOffset > 10) {{
                    navBar.classList.add('scrolled');
                }} else {{
                    navBar.classList.remove('scrolled');
                }}
            }}

            // Scroll event listeners with throttling
            let ticking = false;
            window.addEventListener('scroll', function() {{
                if (!ticking) {{
                    window.requestAnimationFrame(function() {{
                        updateActiveNav();
                        updateNavBarStyle();
                        ticking = false;
                    }});
                    ticking = true;
                }}
            }}, {{ passive: true }});

            // Initial update
            updateActiveNav();
            updateNavBarStyle();
        }}

        document.addEventListener('DOMContentLoaded', renderMarkdown);
    </script>
</body>
</html>"""

    logger.info("HTML report generated successfully")
    return {"html_report": html_content}


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


def route_after_clarify(state: ResearchState):
    clarification_needed = state["clarification_needed"]

    if clarification_needed:
        allow_console_input = state.get("allow_console_input", True)
        if state.get("user_answers") or (allow_console_input and sys.stdin.isatty()):
            next_node = "collect_user_response"
        else:
            next_node = END
            logger.info(
                "Clarification needed in Web UI mode. Stopping graph to wait for user input."
            )
    else:
        next_node = "background_search"

    logger.info(
        f"Routing decision: Clarification needed = {clarification_needed}, Next node = {next_node}"
    )
    return next_node


# Build the graph
builder = StateGraph(ResearchState)
builder.add_node("clarify_query", clarify_query_node)
builder.add_node("background_search", background_search_node)
builder.add_node("generate_keywords", generate_keywords_node)
builder.add_node("multi_search", multi_search_node)
builder.add_node("check_gaps", check_gaps_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("generate_html", generate_html_node)
builder.add_node("collect_user_response", collect_user_response_node)

builder.add_edge(START, "clarify_query")
builder.add_conditional_edges("clarify_query", route_after_clarify)
builder.add_edge("collect_user_response", "background_search")
builder.add_edge("background_search", "generate_keywords")
builder.add_edge("generate_keywords", "multi_search")
builder.add_edge("multi_search", "check_gaps")
builder.add_conditional_edges("check_gaps", route_after_check)
builder.add_edge("synthesize", "generate_html")
builder.add_edge("generate_html", END)

graph = builder.compile()


# Function to run the agent
def run_research_agent(
    query: str,
    max_iterations: int = 3,
    lang: str = "en",
    char_limits: Optional[Dict[str, int]] = None,
    user_answers: Optional[Dict[str, str]] = None,
    allow_console_input: bool = True,
):
    global llm_instances
    if not llm_instances:
        model_config = load_model_config()
        llm_instances_global = create_llm_instances(model_config)
        llm_instances.update(llm_instances_global)
        logger.info("LLM instances initialized")

    logger.info("=" * 50)
    logger.info("RESEARCH AGENT STARTED")
    logger.info(f"Query: {query}")
    logger.info(f"User answers provided: {user_answers is not None}")
    logger.info(f"Console input allowed: {allow_console_input}")
    logger.info("=" * 50)

    if char_limits is None:
        char_limits = {"background": 300, "keyword_summary": 500, "final_report": 2000}

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
        "gap_details": "",
        "final_report": "",
        "current_time": current_time,
        "iteration": 0,
        "max_iterations": max_iterations,
        "keyword_history": [],
        "lang": lang,
        "llm_outputs": {},
        "html_report": "",
        "char_limits": char_limits,
        "background_sources": [],
        "keyword_search_sources": {},
        "clarification_needed": False,
        "user_context": "",
        "clarification_questions": [],
        "user_provided_context": "",
        "user_answers": user_answers,
        "allow_console_input": allow_console_input,
    }

    logger.debug("Initial state prepared, starting graph execution")
    result = graph.invoke(initial_state)

    logger.info("=" * 50)
    logger.info(f"RESEARCH AGENT COMPLETED")
    logger.info(f"Final report length: {len(result['final_report'])} characters")
    logger.info("=" * 50)

    return result


def stream_research_agent(
    query: str,
    max_iterations: int = 3,
    lang: str = "en",
    char_limits: Optional[Dict[str, int]] = None,
    user_answers: Optional[Dict[str, str]] = None,
    allow_console_input: bool = True,
):
    """Stream research agent results node by node for real-time UI updates.

    Yields node updates as they complete, allowing UI to display progress in real-time.

    Yields:
        dict: Contains keys:
            - 'node': Name of completed node
            - 'state': Current state after node execution
            - 'output': Output from the node
    """
    global llm_instances
    if not llm_instances:
        model_config = load_model_config()
        llm_instances_global = create_llm_instances(model_config)
        llm_instances.update(llm_instances_global)
        logger.info("LLM instances initialized")

    logger.info("=" * 50)
    logger.info("RESEARCH AGENT STREAM STARTED")
    logger.info(f"Query: {query}")
    logger.info(f"User answers provided: {user_answers is not None}")
    logger.info(f"Console input allowed: {allow_console_input}")
    logger.info("=" * 50)

    if char_limits is None:
        char_limits = {"background": 300, "keyword_summary": 500, "final_report": 2000}

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
        "gap_details": "",
        "final_report": "",
        "current_time": current_time,
        "iteration": 0,
        "max_iterations": max_iterations,
        "keyword_history": [],
        "lang": lang,
        "llm_outputs": {},
        "html_report": "",
        "char_limits": char_limits,
        "background_sources": [],
        "keyword_search_sources": {},
        "clarification_needed": False,
        "user_context": "",
        "clarification_questions": [],
        "user_provided_context": "",
        "user_answers": user_answers,
        "allow_console_input": allow_console_input,
    }

    logger.debug("Initial state prepared, starting graph streaming")
    state = initial_state.copy()

    for event in graph.stream(state):
        for node_name, node_output in event.items():
            logger.info(f"--- Node completed: {node_name} ---")

            if isinstance(node_output, dict):
                for key, value in node_output.items():
                    state[key] = value

            yield {
                "node": node_name,
                "state": state,
                "output": node_output,
            }

    logger.info("=" * 50)
    logger.info("RESEARCH AGENT STREAM COMPLETED")
    logger.info(f"Final report length: {len(state['final_report'])} characters")
    logger.info("=" * 50)

    yield {
        "node": "complete",
        "state": state,
        "output": None,
    }


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

    parser.add_argument(
        "--limit-background",
        type=int,
        default=None,
        help="Target character limit for background summary (default: 300)",
    )
    parser.add_argument(
        "--limit-keyword",
        type=int,
        default=None,
        help="Target character limit for keyword summaries (default: 500)",
    )
    parser.add_argument(
        "--limit-final",
        type=int,
        default=None,
        help="Target character limit for final report (default: 2000)",
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

    cli_limits = {}
    if args.limit_background is not None:
        cli_limits["background"] = args.limit_background
    if args.limit_keyword is not None:
        cli_limits["keyword"] = args.limit_keyword
    if args.limit_final is not None:
        cli_limits["final"] = args.limit_final

    # Initialize LLM instances with configuration
    model_config = load_model_config(cli_overrides, cli_limits)
    llm_instances_global = create_llm_instances(model_config)
    llm_instances.update(llm_instances_global)

    char_limits = {
        "background": model_config.char_limits.background,
        "keyword_summary": model_config.char_limits.keyword_summary,
        "final_report": model_config.char_limits.final_report,
    }

    # Generate output filename if not provided
    output_file = args.output
    output_dir = None

    if not output_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r"[^\w\s-]", "", args.query)[:30].strip()
        safe_query = re.sub(r"\s+", "_", safe_query)
        ext = "md" if args.format == "markdown" else args.format.lower()

        if args.format == "html":
            output_dir = Path(f"output/{safe_query}_{timestamp}/")
            output_file = str(output_dir / "report.html")
        else:
            output_file = f"output/{safe_query}_{timestamp}.{ext}"
    else:
        if args.format == "html" and Path(output_file).suffix == ".html":
            output_dir = Path(output_file).parent
        else:
            output_dir = None

    # Ensure output directory exists
    output_path = Path(output_file)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ensured: {output_path.parent}")

    # Run research agent
    logger.info(f"Running research agent with query: {args.query}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Output file: {output_file}")

    try:
        report = run_research_agent(
            args.query,
            max_iterations=args.max_iterations,
            lang=args.lang,
            char_limits=char_limits,
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

        if args.format == "html" and output_dir:
            save_node_markdown_files(report, output_dir)
            logger.info(f"✅ Node markdown files saved to: {output_dir}")
    except IOError as e:
        logger.error(f"❌ Failed to write output file: {e}")
        raise

    # Also print to console for immediate feedback
    if args.format == "html":
        if output_dir:
            print(f"HTML report generated at: {output_file}")
            print(f"Node markdown files saved to: {output_dir}")
        else:
            print("HTML report generated. Open the file in a browser to view it.")
    else:
        print(report["final_report"])
    logger.info("Research agent execution finished")
