import os
import json
from typing import TypedDict, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from .embeddings import search


# ============================================================
# MOCK_LLM
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# Pydantic response schema
# ============================================================

class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LangGraph state
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    context: str
    retrieved_ids: list[str]
    response: dict


# ============================================================
# Required structured prompt
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are Zepto's customer support policy assistant.

CONTEXT:
{context}

TASK:
Answer the customer's question using only the Zepto policy
information provided in the context.

FORMAT:
Return valid JSON with exactly these fields:
answer, sources, confidence.

LENGTH:
Keep the answer concise and customer-friendly, using at most
3 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the
provided context. Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:

Question:
Can I cancel my order after it has been packed?

Context:
Orders can be cancelled free of cost any time before the order
status changes to 'Packed'. Once an order has been packed, it can
no longer be cancelled through the app.

Answer:
{
  "answer": "Orders can be cancelled before the order status changes to Packed. Once the order has been packed, it cannot be cancelled through the app.",
  "sources": ["doc_05"],
  "confidence": 1.0
}

Customer Question:
{query}
"""


# ============================================================
# Optional real LLM helper
# ============================================================

def get_real_llm():
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    return ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        ),
        temperature=0
    )


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def real_llm_structured_answer(
    prompt: str,
    fallback_sources: list[str]
) -> SupportResponse:

    last_error: Optional[str] = None

    # Initial attempt + 2 retries
    for attempt in range(3):
        try:
            llm = get_real_llm()

            correction = ""

            if attempt > 0:
                correction = """
IMPORTANT CORRECTION:
The previous response failed schema validation.
Return ONLY valid JSON with:
- answer: string
- sources: list of strings
- confidence: number from 0 to 1
Do not include Markdown or any extra text.
"""

            response = llm.invoke(
                prompt + correction
            )

            raw_text = clean_json_text(response.content)
            parsed = json.loads(raw_text)

            return SupportResponse.model_validate(parsed)

        except Exception as exc:
            last_error = str(exc)

    return SupportResponse(
        answer=(
            "ERROR: Unable to produce a valid structured response. "
            f"{last_error}"
        ),
        sources=fallback_sources,
        confidence=0.0
    )


# ============================================================
# NODE 1 — classify_intent
# ============================================================

def classify_intent(state: GraphState) -> GraphState:

    query = state["query"]
    lower_query = query.lower()

    if MOCK_LLM:

        policy_keywords = [
            "delivery",
            "return",
            "refund",
            "membership",
            "tracking",
            "cancel",
            "gift card",
            "support hours"
        ]

        if any(
            keyword in lower_query
            for keyword in policy_keywords
        ):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:

        llm = get_real_llm()

        prompt = f"""
Classify this customer query into exactly one category:

policy_question
general_question

Query:
{query}

Return only the category name.
"""

        response = llm.invoke(prompt)
        result = response.content.strip().lower()

        if "policy_question" in result:
            intent = "policy_question"
        else:
            intent = "general_question"

    state["intent"] = intent
    return state


# ============================================================
# NODE 2 — retrieve_and_answer
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:

    query = state["query"]

    # Retrieval always runs in both modes
    results = search(query, n_results=3)

    retrieved_ids = results["ids"][0]
    retrieved_documents = results["documents"][0]

    top_chunk = retrieved_documents[0]

    state["retrieved_ids"] = retrieved_ids
    state["context"] = "\n\n".join(retrieved_documents)

    if MOCK_LLM:

        snippet = top_chunk[:200]

        response = SupportResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=retrieved_ids,
            confidence=1.0
        )

    else:

        prompt = PROMPT_TEMPLATE.format(
            context=state["context"],
            query=query
        )

        response = real_llm_structured_answer(
            prompt,
            retrieved_ids
        )

    state["response"] = response.model_dump()

    return state


# ============================================================
# NODE 3 — direct_answer
# ============================================================

def direct_answer(state: GraphState) -> GraphState:

    query = state["query"]

    if MOCK_LLM:

        response = SupportResponse(
            answer=(
                "I can only answer questions about Zepto "
                "policies right now."
            ),
            sources=[],
            confidence=1.0
        )

    else:

        prompt = f"""
ROLE:
You are Zepto's customer support assistant.

TASK:
Answer the following general question clearly.

NEGATIVE CONSTRAINT:
Do not invent Zepto-specific policies.

FORMAT:
Return valid JSON with:
answer, sources, confidence.

LENGTH:
Maximum 3 sentences.

Question:
{query}
"""

        response = real_llm_structured_answer(
            prompt,
            []
        )

    state["response"] = response.model_dump()

    return state


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_classification(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(GraphState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()


# ============================================================
# PUBLIC FUNCTION USED BY FASTAPI
# ============================================================

def ask_question(query: str) -> SupportResponse:

    result = graph.invoke({
        "query": query
    })

    return SupportResponse.model_validate(
        result["response"]
    )
