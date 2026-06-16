import json
import operator
from typing import Annotated, List, Dict, TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import load_config
from src.agents import (
    get_llm, DESIGNER_SYSTEM_PROMPT, CHALLENGER_PROMPTS, CONVERGENCE_MARKER
)
from src.rag import search_knowledge, format_knowledge_context

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class AgentState(TypedDict):
    requirement: str
    challenge_level: str
    current_round: int
    design_doc: str
    conversation: Annotated[List[dict], operator.add]
    converged: bool
    human_review_pending: bool
    skip_initial_generation: bool
    status: str
    token_usage: Dict[str, int]


def _accumulate_tokens(prev: dict, new_tokens: dict) -> dict:
    return {
        "input_tokens": prev.get("input_tokens", 0) + new_tokens.get("input_tokens", 0),
        "output_tokens": prev.get("output_tokens", 0) + new_tokens.get("output_tokens", 0),
        "total_tokens": prev.get("total_tokens", 0) + new_tokens.get("total_tokens", 0),
    }


def _extract_token_usage(response) -> dict:
    meta = response.response_metadata or {}
    usage = meta.get("token_usage", {})
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _build_knowledge_section(requirement: str, design_doc: str) -> str:
    search_query = f"{requirement}\n{design_doc[:2000]}"
    results = search_knowledge(search_query)
    return format_knowledge_context(results)


async def _designer_generate_impl(state: AgentState) -> dict:
    llm = get_llm()
    requirement = state["requirement"]
    knowledge = _build_knowledge_section(requirement, "")

    messages = [
        SystemMessage(content=DESIGNER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Please create a comprehensive design document for the following requirement:

{requirement}

{knowledge}
Generate a complete design document covering all aspects. Use the knowledge references above if relevant. Output in markdown format directly, no preamble.""")
    ]

    response = await llm.ainvoke(messages)
    design_doc = response.content

    new_usage = _extract_token_usage(response)
    prev_usage = state.get("token_usage", {})
    total_usage = _accumulate_tokens(prev_usage, new_usage)

    conv_msg = {
        "role": "designer",
        "content": f"## Initial Design Document\n\n{design_doc}",
        "round": 0,
        "type": "initial"
    }

    return {
        "design_doc": design_doc,
        "conversation": [conv_msg],
        "current_round": 1,
        "token_usage": total_usage,
    }


async def designer_generate_node(state: AgentState) -> dict:
    if state.get("skip_initial_generation"):
        return await designer_respond_node(state)
    return await _designer_generate_impl(state)


async def challenger_review_node(state: AgentState) -> dict:
    llm = get_llm()

    level = state["challenge_level"]
    design_doc = state["design_doc"]
    current_round = state["current_round"]
    requirement = state["requirement"]

    knowledge = _build_knowledge_section(requirement, design_doc)
    system_prompt = CHALLENGER_PROMPTS.get(level, CHALLENGER_PROMPTS["medium"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Review Round {current_round}

Original Requirement:
{requirement}

Current Design Document:
{design_doc}

{knowledge}
Please review and provide your challenges. Use the knowledge references if relevant. If you have no more concerns, sign off with exactly: 我sign off，没有更多疑问""")
    ]

    response = await llm.ainvoke(messages)
    content = response.content

    converged = CONVERGENCE_MARKER in content

    new_usage = _extract_token_usage(response)
    prev_usage = state.get("token_usage", {})
    total_usage = _accumulate_tokens(prev_usage, new_usage)

    conv_msg = {
        "role": "challenger",
        "content": content,
        "round": current_round,
        "type": "challenge"
    }

    result = {
        "conversation": [conv_msg],
        "converged": converged,
        "token_usage": total_usage,
    }

    if converged:
        result["human_review_pending"] = True

    return result


async def designer_respond_node(state: AgentState) -> dict:
    llm = get_llm()

    design_doc = state["design_doc"]
    current_round = state["current_round"]
    requirement = state["requirement"]

    last_challenges = ""
    for msg in reversed(state.get("conversation", [])):
        if msg.get("role") == "challenger":
            last_challenges = msg["content"]
            break

    knowledge = _build_knowledge_section(requirement, design_doc)

    messages = [
        SystemMessage(content=DESIGNER_SYSTEM_PROMPT),
        HumanMessage(content=f"""## Challenges Received (Round {current_round})

{last_challenges}

## Current Design Document

{design_doc}

## Original Requirement

{requirement}

{knowledge}
Please address each challenge point by point, then provide the COMPLETE updated design document.

Format:
## Response to Challenges
[Point-by-point responses]

## Updated Design Document
[Complete updated design document]""")
    ]

    response = await llm.ainvoke(messages)
    content = response.content

    updated_design = content
    if "## Updated Design Document" in content:
        parts = content.split("## Updated Design Document", 1)
        updated_design = parts[1].strip() if len(parts) > 1 else content

    new_usage = _extract_token_usage(response)
    prev_usage = state.get("token_usage", {})
    total_usage = _accumulate_tokens(prev_usage, new_usage)

    conv_msg = {
        "role": "designer",
        "content": content,
        "round": current_round,
        "type": "response"
    }

    return {
        "design_doc": updated_design,
        "conversation": [conv_msg],
        "current_round": current_round + 1,
        "token_usage": total_usage,
    }


def should_continue(state: AgentState) -> str:
    config = load_config()
    safety_max_rounds = config.get("challenge", {}).get("safety_max_rounds", 20)

    if state.get("converged", False):
        return "end"
    if state["current_round"] > safety_max_rounds:
        return "end"
    return "continue"


async def run_human_challenge_loop(state: dict) -> list:
    """Run designer_respond -> challenger_review loop for human-provided challenges.
    Modifies state in-place. Returns list of new conversation messages."""
    new_messages = []

    while True:
        updates = await designer_respond_node(state)
        new_messages.extend(updates.get("conversation", []))
        state["design_doc"] = updates.get("design_doc", state["design_doc"])
        state["current_round"] = updates.get("current_round", state["current_round"])
        state["token_usage"] = updates.get("token_usage", state.get("token_usage", {}))

        updates = await challenger_review_node(state)
        new_messages.extend(updates.get("conversation", []))
        state["converged"] = updates.get("converged", False)
        state["human_review_pending"] = updates.get("human_review_pending", False)
        state["token_usage"] = updates.get("token_usage", state.get("token_usage", {}))

        if state.get("converged"):
            break

        config = load_config()
        safety_max = config.get("challenge", {}).get("safety_max_rounds", 20)
        if state["current_round"] > safety_max:
            break

    return new_messages


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("designer_generate", designer_generate_node)
    builder.add_node("challenger_review", challenger_review_node)
    builder.add_node("designer_respond", designer_respond_node)

    builder.set_entry_point("designer_generate")

    builder.add_edge("designer_generate", "challenger_review")
    builder.add_conditional_edges(
        "challenger_review",
        should_continue,
        {
            "continue": "designer_respond",
            "end": END,
        }
    )
    builder.add_edge("designer_respond", "challenger_review")

    return builder


_graph = None
_checkpointer_cm = None


def get_compiled_graph():
    global _graph, _checkpointer_cm
    if _graph is None:
        _checkpointer_cm = SqliteSaver.from_conn_string(str(DATA_DIR / "checkpoints.db"))
        checkpointer = _checkpointer_cm.__enter__()
        builder = build_graph()
        _graph = builder.compile(checkpointer=checkpointer)
    return _graph


def cleanup_checkpointer():
    global _checkpointer_cm, _graph
    if _checkpointer_cm is not None:
        _checkpointer_cm.__exit__(None, None, None)
        _checkpointer_cm = None
    _graph = None
