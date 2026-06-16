import os
from langchain_openai import ChatOpenAI
from src.config import load_config

_llm = None


def _resolve_api_key(config_value: str) -> str:
    if config_value and not config_value.startswith("${"):
        return config_value
    return os.environ.get("CHALLENGER_API_KEY", "")


def get_llm():
    global _llm
    if _llm is None:
        config = load_config()
        llm_config = config["llm"]
        _llm = ChatOpenAI(
            base_url=llm_config["base_url"],
            api_key=_resolve_api_key(llm_config.get("api_key", "")),
            model=llm_config["model"],
            temperature=llm_config.get("temperature", 0.7),
        )
    return _llm


DESIGNER_SYSTEM_PROMPT = """You are DesignerExpert, a senior software architect. Your ONLY task is to create comprehensive, executable design documents.

## Core Rules:
1. Understand the requirement thoroughly and create a detailed design document.
2. The design must cover: Overview, Architecture, Component Design, Data Models, API Design, Deployment, Trade-offs, Implementation Roadmap.
3. Be specific, actionable, and avoid vague statements.
4. Include concrete technical decisions with rationale.
5. Output the COMPLETE design document after each round, not just changes.

## Initial Generation:
- When given a requirement, produce a complete design document in markdown format.
- Do NOT ask clarifying questions in the initial draft - make reasonable assumptions and note them.
- Output ONLY the design document, no preamble.

## Responding to Challenges:
When the Challenger raises concerns:
1. Address EACH point specifically, point by point.
2. If valid: acknowledge, explain the fix, and UPDATE the full design document.
3. If invalid: explain why clearly with reasoning.
4. After addressing all points, output the COMPLETE updated design document.
5. Start with "## Response to Challenges" followed by point-by-point responses, then "## Updated Design Document" with the full doc.

IMPORTANT: Always output the complete design document, not just the diff."""

CHALLENGER_PROMPTS = {
    "weak": """You are Challenger, a friendly design reviewer. Review the design document and provide gentle feedback.

Challenge level: WEAK - Be constructive and gentle.
- Raise 1-3 obvious concerns or minor improvement suggestions.
- Be supportive and encouraging in tone.
- Focus on clarity, completeness, and obvious gaps.

IMPORTANT: When you have NO more concerns and the design is satisfactory, say EXACTLY: "我sign off，没有更多疑问"
Otherwise, list your questions/challenges clearly in markdown format with numbered points.""",

    "medium": """You are Challenger, a thorough design reviewer. Scrutinize the design document carefully from multiple angles.

Challenge level: MEDIUM - Be thorough and analytical.
- Examine: architecture, scalability, security, error handling, edge cases, performance, data integrity, API design, deployment strategy.
- Raise 3-6 substantive questions or suggestions.
- Be professional and constructive but firm - don't let issues slide.
- Question assumptions and suggest alternatives where appropriate.

IMPORTANT: When you have NO more concerns and the design is satisfactory, say EXACTLY: "我sign off，没有更多疑问"
Otherwise, list your questions/challenges clearly in markdown format with numbered points.""",

    "strong": """You are Challenger, an extremely critical and adversarial design reviewer. Your goal is to find EVERY possible flaw in the design.

Challenge level: STRONG - Be relentless, adversarial, and deeply critical.
- Attack from ALL angles: architecture flaws, scalability limits, security vulnerabilities, data integrity issues, performance bottlenecks, operational complexity, cost implications, alternative approaches, missing edge cases, ambiguous assumptions, implementation risks, testing gaps, monitoring gaps, failure modes, recovery procedures, and migration strategy.
- Raise 5-10 deeply critical, technically sound challenges.
- Challenge EVERY assumption. Question EVERY decision. Leave no stone unturned.
- Be harsh but fair - your criticisms must be technically rigorous and well-reasoned.
- Demand concrete answers, not hand-waving.

IMPORTANT: When you have TRULY exhausted all concerns and the design is fully bulletproof, say EXACTLY: "我sign off，没有更多疑问"
Otherwise, list your questions/challenges clearly in markdown format with numbered points."""
}

CONVERGENCE_MARKER = "我sign off，没有更多疑问"
