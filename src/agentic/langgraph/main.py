from langgraph.graph import StateGraph

from src.agentic.langgraph.chunk_graph_factory import create_chunk_ranking_graph
from src.agentic.langgraph.doc_graph import create_document_ranking_graph


def initialize_langgraph_workflows(agentic_version: int) -> tuple[StateGraph, StateGraph]:
    """Initialize LangGraph workflows for document and chunk ranking.

    Creates and compiles both document ranking and chunk ranking workflows
    for use in the multi-agent evaluation pipeline.

    Returns:
        tuple[StateGraph, StateGraph]: A tuple containing:
            - document_ranking_graph: Compiled workflow for document type ranking
            - chunk_ranking_graph: Compiled workflow for text chunk ranking

    Note:
        Both graphs are compiled and ready for execution via ainvoke().
    """
    document_ranking_graph = create_document_ranking_graph()
    chunk_ranking_graph = create_chunk_ranking_graph(agentic_version)

    return document_ranking_graph, chunk_ranking_graph
