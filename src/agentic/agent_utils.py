from src.schema import AgentWeights


async def convert_agent_weights_to_dict(raw_agent_weights: AgentWeights | dict) -> dict[str, float]:
    """Convert AgentWeights object to a standardized dictionary format.

    This function handles the conversion of various agent weight formats into a
    standardized dictionary with string keys and float values. It supports
    AgentWeights objects, dictionaries, and provides fallback defaults.

    Args:
        raw_agent_weights (AgentWeights | dict): The agent weights in various formats
            (AgentWeights class or dictionary).

    Returns:
        dict[str, float]: A dictionary mapping document type names to their
            corresponding weight values. Keys are: "DEF14A", "10-K", "10-Q",
            "8-K", "Earnings". All values sum to 1.0.

    Note:
        If conversion fails, returns default weights of 0.2 for each document type.
    """
    if hasattr(raw_agent_weights, "to_dict"):
        # It's an AgentWeights object
        return raw_agent_weights.to_dict()
    if hasattr(raw_agent_weights, "__dict__"):
        # It's an AgentWeights object without to_dict method
        return {
            "DEF14A": raw_agent_weights.DEF14A_agent,
            "10-K": raw_agent_weights.tenK_agent,
            "10-Q": raw_agent_weights.tenQ_agent,
            "8-K": raw_agent_weights.eightK_agent,
            "Earnings": raw_agent_weights.Earnings_agent,
        }
    if isinstance(raw_agent_weights, dict):
        # It's already a dictionary
        return raw_agent_weights
    print("❌ Error occurred while converting agent weights, using default weights...")
    # Default weights if nothing is available
    return {"DEF14A": 0.2, "10-K": 0.2, "10-Q": 0.2, "8-K": 0.2, "Earnings": 0.2}
