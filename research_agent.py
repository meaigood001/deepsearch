import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated
import operator
import datetime

load_dotenv()

def get_time_awareness_instruction(current_time: str, lang: str = "en") -> str:
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
"""
    }
    return instructions.get(lang, instructions["en"])

def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Define the state
class ResearchState(TypedDict):
    query: str
    original_query: str
    background: str
    confirmed: bool
    keywords: Annotated[list, operator.add]
    summaries: Annotated[list, operator.add]
    gaps_found: bool
    final_report: str
    current_time: str


# Define tools
@tool
def web_search(query: str) -> str:
    """Search the web for information related to the query."""
    search = TavilySearchResults(max_results=5)
    results = search.invoke({"query": query})
    return str(results)


@tool
def analyze_insight(info: str) -> str:
    """Analyze and extract key insights from information."""
    return f"Key insights from: {info}"


# Define nodes
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def background_search_node(state: ResearchState):
    query = state["query"]
    search_tool = web_search
    background = search_tool.invoke(query)
    return {
        "background": background,
        "confirmed": True,
    }  # Assume auto-confirm for simplicity


def generate_keywords_node(state: ResearchState):
    background = state["background"]
    time_instruction = get_time_awareness_instruction(state["current_time"])
    original_query = state["original_query"]
    prompt = ChatPromptTemplate.from_template(
        f"{time_instruction}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: Your search should help answer their specific question, not just collect general information about the paragraph topic.\nBased on the background: {{background}}, generate multiple groups of keywords for in-depth research on: {{query}}"
    )
    chain = prompt | llm | StrOutputParser()
    keywords_str = chain.invoke({"background": background, "query": state["query"]})
    keywords = [kw.strip() for kw in keywords_str.split(",")]
    return {"keywords": keywords}


def multi_search_node(state: ResearchState):
    keywords = state["keywords"]
    summaries = []
    time_instruction = get_time_awareness_instruction(state["current_time"])
    original_query = state["original_query"]
    for kw in keywords:
        search_tool = web_search
        result = search_tool.invoke(kw)
        # Summarize
        prompt = ChatPromptTemplate.from_template(
            f"{time_instruction}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: The paragraph must help answer the user's original research question. Focus on information relevant to their specific inquiry.\nSummarize the following search results for keyword '{{kw}}': {{result}}"
        )
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"kw": kw, "result": result})
        summaries.append(summary)
    return {"summaries": summaries}


def check_gaps_node(state: ResearchState):
    summaries = state["summaries"]
    query = state["query"]
    time_instruction = get_time_awareness_instruction(state["current_time"])
    original_query = state["original_query"]
    prompt = ChatPromptTemplate.from_template(
        f"{time_instruction}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: Reflect on whether the current paragraphs sufficiently address the user's original research question. Identify deficiencies, especially in answering the query.\nReview these summaries: {{summaries}}\nAgainst the query: {{query}}\nAre there gaps? Answer 'yes' or 'no'."
    )
    chain = prompt | llm | StrOutputParser()
    gaps = chain.invoke({"summaries": "\n".join(summaries), "query": query})
    gaps_found = "yes" in gaps.lower()
    return {"gaps_found": gaps_found}


def synthesize_node(state: ResearchState):
    summaries = state["summaries"]
    query = state["query"]
    time_instruction = get_time_awareness_instruction(state["current_time"])
    original_query = state["original_query"]
    prompt = ChatPromptTemplate.from_template(
        f"{time_instruction}\n🎯 USER'S ORIGINAL REQUEST: {original_query}\n💡 IMPORTANT: Ensure the final report comprehensively answers the user's original research question. Conclude by summarizing how the report addresses their specific inquiry.\nSynthesize these summaries into a comprehensive report for the query: {{query}}\nSummaries: {{summaries}}"
    )
    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({"query": query, "summaries": "\n".join(summaries)})
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
    return "generate_keywords" if state["gaps_found"] else "synthesize"


builder.add_conditional_edges("check_gaps", route_after_check)
builder.add_edge("synthesize", END)

graph = builder.compile()


# Function to run the agent
def run_research_agent(query: str):
    current_time = get_current_time()
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
    }
    result = graph.invoke(initial_state)
    return result["final_report"]


if __name__ == "__main__":
    query = "What are the latest developments in AI safety?"
    report = run_research_agent(query)
    print(report)
