from langgraph.graph import END, StateGraph

from src.agentic.chunk.chunk_state_factory import (
    ChunkRankingStateV1,
    ChunkRankingStateV2,
    ChunkRankingStateV3,
    ChunkRankingStateV4,
)
from src.agentic.chunk.utils.v1 import build_chunk_consensus_v1, evaluate_chunks_parallel_v1, facilitate_discussion_v1
from src.agentic.chunk.utils.v2 import build_chunk_consensus_v2, evaluate_chunks_parallel_v2, facilitate_discussion_v2
from src.agentic.chunk.utils.v3 import (
    build_consensus_ranking_v3,
    facilitate_discussion_v3,
    stage1_quick_filter,
    stage2_deep_scoring,
)
from src.agentic.chunk.utils.v4 import build_chunk_consensus_v4, evaluate_chunks_parallel_v4, facilitate_discussion_v4


def create_chunk_ranking_graph(agentic_version: int) -> StateGraph:
    """Create the chunk ranking workflow graph using LangGraph for specified version.

    Builds a multi-agent workflow for chunk ranking that coordinates specialized
    agents to evaluate text chunks and build consensus rankings. The workflow
    structure varies based on the agentic version, enabling different agent
    architectures and evaluation strategies.

    Args:
        agentic_version (int): Version of the agentic workflow to create:
            - Version 1: Multi-role agent evaluation with discussion and consensus
            - Version 2: Phased evaluation with noise removal, candidate selection,
              and deep scoring agents
            - Version 3: Two-stage pipeline with quick filtering followed by deep
              scoring with specialized reasoning agents
            - Version 4: Simplified multi-agent evaluation with financial and risk
              analyst perspectives

    Returns:
        StateGraph: Compiled LangGraph workflow configured for the specified
            agentic version. The graph includes nodes for chunk evaluation,
            discussion facilitation, and consensus building appropriate to
            the chosen version.

    Note:
        Each version uses a different state type (ChunkRankingStateV1-V4) with
        version-specific agent configurations and workflow patterns optimized
        for different ranking strategies.
    """
    if agentic_version == 1:
        graph = StateGraph(ChunkRankingStateV1)

        # Add nodes
        graph.add_node("evaluate_chunks", evaluate_chunks_parallel_v1)
        graph.add_node("facilitate_discussion", facilitate_discussion_v1)
        graph.add_node("build_consensus", build_chunk_consensus_v1)

        # Add edges
        graph.add_edge("evaluate_chunks", "facilitate_discussion")
        graph.add_edge("facilitate_discussion", "build_consensus")
        graph.add_edge("build_consensus", END)

        # Set entry point
        graph.set_entry_point("evaluate_chunks")
    elif agentic_version == 2:
        graph = StateGraph(ChunkRankingStateV2)

        # Add nodes
        graph.add_node("evaluate_chunks", evaluate_chunks_parallel_v2)
        graph.add_node("facilitate_discussion", facilitate_discussion_v2)
        graph.add_node("build_consensus", build_chunk_consensus_v2)

        # Add edges
        graph.add_edge("evaluate_chunks", "facilitate_discussion")
        graph.add_edge("facilitate_discussion", "build_consensus")
        graph.add_edge("build_consensus", END)

        # Set entry point
        graph.set_entry_point("evaluate_chunks")
    elif agentic_version == 3:
        graph = StateGraph(ChunkRankingStateV3)

        # Add nodes
        graph.add_node("stage1_filter", stage1_quick_filter)
        graph.add_node("stage2_scoring", stage2_deep_scoring)
        graph.add_node("discussion", facilitate_discussion_v3)
        graph.add_node("consensus", build_consensus_ranking_v3)

        # Add edges
        graph.add_edge("stage1_filter", "stage2_scoring")
        graph.add_edge("stage2_scoring", "discussion")
        graph.add_edge("discussion", "consensus")
        graph.add_edge("consensus", END)

        # Set entry point
        graph.set_entry_point("stage1_filter")
    else:
        graph = StateGraph(ChunkRankingStateV4)

        # Add nodes
        graph.add_node("evaluate_chunks", evaluate_chunks_parallel_v4)
        graph.add_node("facilitate_discussion", facilitate_discussion_v4)
        graph.add_node("build_consensus", build_chunk_consensus_v4)

        # Add edges
        graph.add_edge("evaluate_chunks", "facilitate_discussion")
        graph.add_edge("facilitate_discussion", "build_consensus")
        graph.add_edge("build_consensus", END)

        # Set entry point
        graph.set_entry_point("evaluate_chunks")

    return graph.compile()
