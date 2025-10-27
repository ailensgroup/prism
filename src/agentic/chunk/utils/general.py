import re


def parse_chunk_ranking_content(content: str) -> tuple:
    """Parse chunk ranking content to extract question and text chunks.

    Extracts the question and chunk information from raw content using regex
    patterns to identify chunk index blocks and their associated text content.

    Args:
        content (str): Raw content containing question and chunk index blocks
            in the format "[Chunk Index N] chunk_text".

    Returns:
        tuple: A tuple containing (question, chunks, chunk_indices) where:
            - question (str): Extracted question text
            - chunks (list[str]): List of text chunk contents
            - chunk_indices (list[int]): Corresponding original indices for each chunk
    """
    # Extract question
    question_match = re.search(r"Question:\s*(.*?)\n", content)
    question = question_match.group(1).strip() if question_match else ""

    # Extract chunks
    chunk_pattern = r"\[Chunk Index (\d+)\]\s*(.*?)(?=\[Chunk Index|\nTask:|$)"
    matches = re.findall(chunk_pattern, content, re.DOTALL)

    chunks = []
    chunk_indices = []

    for match in matches:
        idx = int(match[0])
        chunk_content = match[1].strip()
        if chunk_content:
            chunks.append(chunk_content)
            chunk_indices.append(idx)

    return question, chunks, chunk_indices
