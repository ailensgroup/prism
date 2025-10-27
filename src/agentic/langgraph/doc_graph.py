from langgraph.graph import END, START, StateGraph

from src.agentic.doc.doc_experts import DocumentRankingState
from src.agentic.doc.doc_utils import (
    build_document_consensus,
    evaluate_documents_parallel,
)


def create_document_ranking_graph() -> StateGraph:
    """Create the document ranking workflow graph using LangGraph.

    Builds a sequential workflow for document ranking that coordinates multiple
    expert agents to evaluate document types and build consensus rankings.

    Returns:
        StateGraph: Compiled LangGraph workflow with nodes for:
            - evaluate_documents: Parallel evaluation by all expert agents
            - build_consensus: Weighted consensus building from agent responses

    Note:
        The graph follows a simple sequential flow: START -> evaluate_documents
        -> build_consensus -> END.
    """
    graph = StateGraph(DocumentRankingState)

    # Add nodes in sequence
    graph.add_node("evaluate_documents", evaluate_documents_parallel)  # This handles all agents
    graph.add_node("build_consensus", build_document_consensus)

    # Sequential flow
    graph.add_edge(START, "evaluate_documents")
    graph.add_edge("evaluate_documents", "build_consensus")
    graph.add_edge("build_consensus", END)

    return graph.compile()
