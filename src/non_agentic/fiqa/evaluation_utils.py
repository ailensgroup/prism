import math


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain at position k.

    Args:
        relevances (List[int]): List of relevance scores in ranked order.
        k (int): Position cutoff.

    Returns:
        float: DCG@k score.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        if i == 0:
            dcg += rel
        else:
            dcg += rel / math.log2(i + 1)
    return dcg


def calculate_ndcg(relevances: list[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at position k.

    Args:
        relevances (List[int]): List of relevance scores in ranked order.
        k (int): Position cutoff.

    Returns:
        float: nDCG@k score between 0 and 1.
    """
    dcg = calculate_dcg(relevances, k)

    # Calculate ideal DCG (sort relevances in descending order)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_recall(retrieved_relevant: int, total_relevant: int) -> float:
    """Calculate Recall.

    Args:
        retrieved_relevant (int): Number of relevant documents retrieved.
        total_relevant (int): Total number of relevant documents.

    Returns:
        float: Recall score between 0 and 1.
    """
    if total_relevant == 0:
        return 0.0
    return retrieved_relevant / total_relevant
