"""Human-in-the-loop sponsor recommendation review.

This small, synthetic workflow ranks sponsors deterministically, pauses at an
explicit review checkpoint, and resumes with a human decision. The
``InMemorySaver`` preserves the graph state between the initial invocation and
``Command(resume=...)``. In a production workflow, the interrupt payload would
be presented to a reviewer and the resume value would come from the review
system instead of this script's deterministic demo decisions.

Human-in-the-loop mapping:

* ``rank_sponsors`` is the automated recommendation step.
* ``request_review`` is the human approval gate created by ``interrupt``.
* ``Command(resume="approve")`` and ``Command(resume="reject")`` model the
  explicit reviewer outcomes.
* ``publish_recommendation`` and ``decline_recommendation`` are separate
  downstream paths, so rejection cannot accidentally publish a sponsor.

Downstream contribution on Troy Bacon's fork: Troy Bacon added this focused
sponsor-review routing example to demonstrate the review checkpoint and its
approval/rejection branches.

Run with ``python examples/human_in_the_loop/sponsor_review.py`` from a
checkout that has the local LangGraph packages installed.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class Sponsor(TypedDict):
    """A synthetic sponsor candidate and its deterministic fit score."""

    name: str
    audience_fit: int
    budget_fit: int
    mission_fit: int


class SponsorReviewState(TypedDict):
    """State carried through ranking, review, and the final routing decision."""

    candidates: list[Sponsor]
    ranked: list[Sponsor]
    recommendation: Sponsor | None
    decision: Literal["approve", "reject"] | None
    outcome: str | None


CANDIDATES: list[Sponsor] = [
    {
        "name": "Northstar Bikes",
        "audience_fit": 9,
        "budget_fit": 7,
        "mission_fit": 8,
    },
    {
        "name": "Harbor Coffee",
        "audience_fit": 8,
        "budget_fit": 9,
        "mission_fit": 6,
    },
    {
        "name": "Juniper Solar",
        "audience_fit": 7,
        "budget_fit": 6,
        "mission_fit": 10,
    },
]


def rank_sponsors(state: SponsorReviewState) -> dict[str, object]:
    """Score and sort candidates using only synthetic, deterministic inputs."""

    ranked = sorted(
        state["candidates"],
        key=lambda sponsor: (
            -(
                sponsor["audience_fit"]
                + sponsor["budget_fit"]
                + sponsor["mission_fit"]
            ),
            sponsor["name"],
        ),
    )
    return {"ranked": ranked, "recommendation": ranked[0]}


def request_review(state: SponsorReviewState) -> dict[str, object]:
    """Pause for a reviewer and record the normalized decision on resume."""

    recommendation = state["recommendation"]
    if recommendation is None:  # Defensive guard for direct node reuse.
        raise ValueError("A recommendation is required before requesting review")

    decision = interrupt(
        {
            "question": "Approve this sponsor recommendation?",
            "recommendation": recommendation["name"],
            "ranked_sponsors": [sponsor["name"] for sponsor in state["ranked"]],
            "allowed_decisions": ["approve", "reject"],
        }
    )
    if decision not in {"approve", "reject"}:
        raise ValueError("Review decision must be 'approve' or 'reject'")
    return {"decision": decision}


def route_review(state: SponsorReviewState) -> str:
    """Select exactly one explicit downstream branch after human review."""

    decision = state["decision"]
    if decision is None:
        raise ValueError("A review decision is required before routing")
    return decision


def publish_recommendation(state: SponsorReviewState) -> dict[str, str]:
    """Represent the approved downstream action."""

    recommendation = state["recommendation"]
    if recommendation is None:
        raise ValueError("Cannot publish without a recommendation")
    return {"outcome": f"published:{recommendation['name']}"}


def decline_recommendation(state: SponsorReviewState) -> dict[str, str]:
    """Represent the rejected downstream action."""

    recommendation = state["recommendation"]
    if recommendation is None:
        raise ValueError("Cannot decline without a recommendation")
    return {"outcome": f"rejected:{recommendation['name']}"}


def build_graph() -> StateGraph:
    """Build the sponsor workflow graph before compiling it with a saver."""

    builder = StateGraph(SponsorReviewState)
    builder.add_node("rank_sponsors", rank_sponsors)
    builder.add_node("request_review", request_review)
    builder.add_node("publish_recommendation", publish_recommendation)
    builder.add_node("decline_recommendation", decline_recommendation)
    builder.add_edge(START, "rank_sponsors")
    builder.add_edge("rank_sponsors", "request_review")
    builder.add_conditional_edges(
        "request_review",
        route_review,
        {
            "approve": "publish_recommendation",
            "reject": "decline_recommendation",
        },
    )
    builder.add_edge("publish_recommendation", END)
    builder.add_edge("decline_recommendation", END)
    return builder


def run_review(decision: Literal["approve", "reject"], thread_id: str) -> SponsorReviewState:
    """Pause once, then resume the same checkpoint with ``decision``."""

    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    paused = graph.invoke(
        {
            "candidates": CANDIDATES,
            "ranked": [],
            "recommendation": None,
            "decision": None,
            "outcome": None,
        },
        config,
    )
    interrupt_payload = paused["__interrupt__"][0].value
    print(f"{thread_id}: review requested for {interrupt_payload['recommendation']}")
    return graph.invoke(Command(resume=decision), config)


def main() -> None:
    """Demonstrate both explicit human-review outcomes."""

    for decision in ("approve", "reject"):
        result = run_review(decision, f"sponsor-review-{decision}")
        print(f"{decision}: {result['outcome']}")


if __name__ == "__main__":
    main()
