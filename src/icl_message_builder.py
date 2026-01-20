import json
import os
import random
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings


class ICLMessageBuilder:
    """Build In-Context Learning (ICL) messages via vector similarity search.

    This helper encapsulates loading/creating a FAISS vector store from
    training data, retrieving nearest examples given a query, and formatting
    the examples as ICL messages.

    Attributes:
        training_data_path (str): Path to the JSONL training dataset.
        icl_n (int): Number of examples to retrieve (post-filtering).
        vector_store_path (str): Directory path where the FAISS index is stored.
        embeddings (AzureOpenAIEmbeddings): Embedding client for indexing/search.
        vector_store (FAISS | None): Loaded/created FAISS store.
        training_data (List[Dict]): Parsed training items (one per line).
    """

    def __init__(
        self,
        training_data_path: str,
        embedding_model: str = "text-embedding-3-small",
        vector_store_path: str = "./embedding",
        document_type: str = "chunk",
        icl_n: int = 5,
        azure_openai_endpoint: str = "dummy_endpoint",
        azure_openai_key: str = "dummy_key",
    ) -> None:
        """Initialize the ICLMessageBuilder and prepare the vector store.

        Args:
            training_data_path (str): Path to the JSONL training data file.
            embedding_model (str, optional): Embedding model name. Defaults to
                `"text-embedding-3-small"`.
            vector_store_path (str, optional): Base directory for the FAISS store.
                A subfolder is created using document type and model name.
                Defaults to `"./embedding"`.
            document_type (str, optional): Logical document type used to construct
                the store path (e.g., `"chunk"`, `"document"`). Defaults to `"chunk"`.
            icl_n (int, optional): Number of ICL examples to return after filtering.
                Defaults to 5.
            azure_openai_endpoint (str, optional): Azure OpenAI endpoint. Defaults to
                `"dummy_endpoint"`.
            azure_openai_key (str, optional): Azure OpenAI API key. Defaults to
                `"dummy_key"`.
        """
        self.training_data_path = training_data_path
        self.icl_n = icl_n
        self.document_type = document_type

        # Construct vector store path
        self.vector_store_path = os.path.join(
            vector_store_path,
            f"icl_store_{document_type}_{embedding_model.replace('-', '_')}",
        )

        self.embeddings = AzureOpenAIEmbeddings(
            api_key=azure_openai_key,
            azure_endpoint=azure_openai_endpoint,
            api_version="2024-02-01",
            model=embedding_model,
            azure_deployment="text-embedding-3-small",
        )

        # Load or create vector store
        self.vector_store = None
        self.training_data = []

        self._initialize_vector_store()

    def _ensure_path_exists(self) -> None:
        """Ensure directories for the vector store exist.

        Creates the main vector store directory and its parent directory if they
        are missing. This method is idempotent.

        Returns:
            None
        """
        if not os.path.exists(self.vector_store_path):
            os.makedirs(self.vector_store_path, exist_ok=True)

        store_dir = os.path.dirname(self.vector_store_path)
        os.makedirs(store_dir, exist_ok=True)

    def _initialize_vector_store(self) -> None:
        """Load an existing FAISS store or create a new one from training data.

        Attempts to load a FAISS index from ``self.vector_store_path``. If loading
        fails or the store does not exist, it loads the training data, prepares
        documents, builds a FAISS index, and saves it to disk.

        Returns:
            None
        """
        print("🔧 Initializing ICL Vector Store...")
        self._ensure_path_exists()

        # Check if vector store exists on vector store path
        if os.path.exists(self.vector_store_path):
            print(f"📂 Loading existing vector store from vector store path: {self.vector_store_path}")
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print("✅ Vector store loaded successfully from vector store path")
                return
            except Exception as e:
                print(f"⚠️ Could not load vector store: {e}")
                print("   Creating new vector store...")

        # Create new vector store from training data
        print(f"📖 Loading training data from {self.training_data_path}")
        self.training_data = self._load_training_data()

        if not self.training_data:
            msg = f"No training data found at {self.training_data_path}"
            raise ValueError(msg)

        print(f"📊 Creating vector store from {len(self.training_data)} examples")
        if self.document_type == "financebench":
            print("📄 Preparing FinanceBench documents for indexing")
            documents = self._prepare_financebench_documents()
        elif self.document_type == "fiqa":
            print("📄 Preparing FiQA documents for indexing")
            documents = self._prepare_fiqa_documents()
        else:
            print("📄 Preparing documents for indexing")
            documents = self._prepare_documents()

        self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)

        # Save vector store to vector store path
        print(f"💾 Saving vector store to vector store path: {self.vector_store_path}")
        self.vector_store.save_local(self.vector_store_path)
        print("✅ Vector store saved to vector store path successfully")

    def _load_training_data(self) -> list[dict]:
        """Load training items from a JSONL file.

        Each line is parsed as a standalone JSON object and appended to the
        resulting list. Errors are printed and an empty list is returned.

        Returns:
            List[Dict]: Parsed items from the JSONL, or an empty list on error.

        Notes:
            - Missing or malformed files do not raise; they are logged and result
              in ``[]`` being returned.
        """
        data = []
        try:
            with open(self.training_data_path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append(item)
        except FileNotFoundError:
            print(f"❌ File not found: {self.training_data_path}")
            return []
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return []
        return data

    def _prepare_documents(self) -> list[Document]:
        """Convert training items to LangChain ``Document`` objects.

        Extracts a question from each training item (from the first message's
        content). Builds a ``Document`` with the question as ``page_content`` and
        metadata including sample identifiers and ground truth labels.

        Returns:
            List[Document]: Prepared documents ready for FAISS indexing.
        """
        documents = []

        for idx, item in enumerate(self.training_data):
            # Extract question from messages
            question = ""
            if "messages" in item and len(item["messages"]) > 0:
                content = item["messages"][0].get("content", "")
                # Parse question from content
                if "Question:" in content:
                    question = content.split("Question:")[1].split("\n")[0].strip()
                else:
                    question = content[:500]  # Use first 500 chars as fallback

            # Extract ground truth ranking from qrel field
            ground_truth = self._extract_ground_truth_from_qrel(item.get("qrel", {}))

            # Create document with metadata
            doc = Document(
                page_content=question,
                metadata={
                    "index": idx,
                    "sample_id": item.get("uuid", item.get("_id", f"sample_{idx}")),
                    "ground_truth": ground_truth,
                    "full_content": item.get("messages", []),
                    "qrel": item.get("qrel", {}),
                },
            )
            documents.append(doc)

        return documents

    def _prepare_financebench_documents(self) -> list[Document]:
        """Convert training items to LangChain ``Document`` objects.

        Extracts a question from each training item (from the first message's
        content). Builds a ``Document`` with the question as ``page_content`` and
        metadata including sample identifiers and ground truth labels.

        Returns:
            List[Document]: Prepared documents ready for FAISS indexing.
        """
        documents = []

        for idx, item in enumerate(self.training_data):

            def safe_str(value: any) -> str:
                if value is None:
                    return ""
                if isinstance(value, (list, dict)):
                    return str(value)
                if isinstance(value, (int, float, bool)):
                    return value
                return str(value).strip()

            metadata = {
                "index": int(idx),
                "sample_id": safe_str(item.get("uuid", item.get("_id", f"sample_{idx}"))),
                "question": safe_str(item.get("question", "")),
                "question_type": safe_str(item.get("question_type", "")),
                "answer": safe_str(item.get("answer", "")),
                "justification": safe_str(item.get("justification", "")),
            }

            # Create document with metadata
            doc = Document(
                page_content=safe_str(item.get("question", "")),
                metadata=metadata,
            )
            documents.append(doc)

        for doc in documents:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)
            # Remove weird characters
            doc.page_content = doc.page_content.replace("\x00", "").strip()

        for i, doc in enumerate(documents[:5]):
            print(f"Doc {i} length: {len(doc.page_content)}, type: {type(doc.page_content)}")

        return documents

    def _prepare_fiqa_documents(self) -> list[Document]:
        """Convert FiQA training items to LangChain Document objects.

        Extracts queries from the training data and their associated relevant documents
        from qrels. Builds Documents with query text as page_content and metadata
        including relevant document IDs and relevance scores.

        Returns:
            List[Document]: Prepared documents ready for FAISS indexing.
        """
        documents = []

        for idx, item in enumerate(self.training_data):
            # Extract query information
            query_id = item.get("query_id", f"query_{idx}")
            query_text = item.get("text", "")

            if not query_text:
                continue

            # Extract relevant documents and scores from qrel
            relevant_docs = item.get("relevant_docs", [])
            relevance_scores = item.get("relevance_scores", {})

            # Create document with metadata
            doc = Document(
                page_content=query_text,
                metadata={
                    "index": idx,
                    "query_id": query_id,
                    "sample_id": query_id,
                    "relevant_docs": relevant_docs,
                    "relevance_scores": relevance_scores,
                },
            )
            documents.append(doc)

        print(f"📊 Prepared {len(documents)} FiQA documents")
        return documents

    def _parse_document_ranking_content(self, content: str) -> tuple:
        """Parse document-ranking content to extract question and document labels.

        Args:
            content (str): Prompt text containing a question and labeled documents.

        Returns:
            tuple: A tuple ``(question, documents, document_indices)`` where
            ``question`` is a string, ``documents`` is a list of document labels/
            types, and ``document_indices`` is a list of corresponding indices
            (ints) parsed from ``[Document Index N]`` headers.

        Notes:
            - The method expects lines like ``[Document Index 3] <Label>``.
        """
        # Extract question
        question_match = re.search(r"Question:\s*(.*?)\n", content)
        question = question_match.group(1).strip() if question_match else ""

        # Extract documents
        doc_pattern = r"\[Document Index (\d+)\]\s*([^\n\[]+)"
        matches = re.findall(doc_pattern, content)

        documents = []
        document_indices = []

        for match in matches:
            idx = int(match[0])
            doc_type = match[1].strip()
            documents.append(doc_type)
            document_indices.append(idx)

        return question, documents, document_indices

    def _parse_chunk_ranking_content(self, content: str) -> tuple:
        """Parse chunk-ranking content to extract question and chunk bodies.

        Args:
            content (str): Prompt text containing a question and labeled chunks.

        Returns:
            tuple: A tuple ``(question, chunks, chunk_indices)`` where
            ``question`` is a string, ``chunks`` is a list of chunk texts, and
            ``chunk_indices`` is a list of indices parsed from headers formatted
            like ``[Chunk Index N]``.

        Notes:
            - Uses a DOTALL regex to capture multi-line chunk content.
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

    def _extract_ground_truth_from_qrel(self, qrel: dict) -> list[int]:
        """Extract ranked ground-truth indices from a qrel mapping.

        Converts a qrel dict (e.g., ``{"0": 0, "1": 2, "25": 2, "32": 1}``) into
        a list of indices sorted by descending relevance score, then ascending
        index for ties. Only indices with positive scores are included. If none
        are positive, the top scored indices (up to 10) are returned.

        Args:
            qrel (Dict): Mapping of string indices to integer relevance scores.

        Returns:
            List[int]: Ranked list of relevant indices.
        """
        if not qrel:
            return []

        # Convert string keys to integers and create list of (index, score) tuples
        scored_items = []
        for idx_str, score in qrel.items():
            try:
                idx = int(idx_str)
                scored_items.append((idx, score))
            except (ValueError, TypeError):
                continue

        # Sort by score (descending), then by index (ascending) for ties
        scored_items.sort(key=lambda x: (-x[1], x[0]))

        # Return only indices with score > 0 (relevant items)
        # Or return all if you want the full ranking
        ground_truth = [idx for idx, score in scored_items if score > 0]

        # If no relevant items, return top items anyway
        if not ground_truth:
            ground_truth = [idx for idx, score in scored_items[:10]]

        return ground_truth

    def get_icl_messages(
        self,
        full_content: dict | None = None,
        include_ground_truth: bool = True,
        format_style: str = "concise",  # "concise" or "detailed"
        doc_type: str = "chunk",  # "chunk" or "document"
        ensure_unique_questions: bool = True,  # Ensure retrieved examples have different questions
        retrieve_k_multiplier: int = 3,  # Multiplier for initial retrieval to account for duplicates
    ) -> list[dict]:
        """Retrieve and format ICL examples for a given query.

        Performs similarity search using the question within ``full_content`` (or
        raises if absent), optionally filters to unique questions, and formats the
        retrieved items as assistant messages. If ground-truth qrels are available,
        top indices are included according to self.icl_n.

        Args:
            full_content (Dict | None): Dictionary containing the prompt
                (``content``) and relevance mapping (``qrel``). Required for
                parsing the question.
            include_ground_truth (bool, optional): Whether to include ground-truth
                indices in the assistant messages. Defaults to True.
            format_style (str, optional): Output style, either ``"concise"`` or
                ``"detailed"``. Defaults to ``"concise"``.
            doc_type (str, optional): Content type for parsing/formatting,
                ``"chunk"`` or ``"document"``. Defaults to ``"chunk"``.
            ensure_unique_questions (bool, optional): If True, filters retrieval
                results to ensure unique question texts. Defaults to True.
            retrieve_k_multiplier (int, optional): Multiplier applied to
                ``icl_n`` for the initial retrieval when uniqueness filtering is
                enabled. Defaults to 3.

        Returns:
            List[Dict] | Tuple[List[Dict], Dict]: A list of assistant messages. If
            internal metadata is assembled (e.g., sentence counts), a tuple is
            returned of ``(messages, metadata)``.
        """
        if not self.vector_store:
            msg = "Vector store not initialized"
            raise ValueError(msg)

        # Handle two input modes: parsed or full_content
        if full_content is not None:
            # Parse from full_content
            content = full_content.get("content", "")
            qrel = full_content.get("qrel", {})

            if doc_type == "document":
                question, sentences, sentence_indices = self._parse_document_ranking_content(content)
            elif doc_type == "chunk":
                question, sentences, sentence_indices = self._parse_chunk_ranking_content(content)
            else:
                msg = f"Unknown doc_type: {doc_type}"
                raise ValueError(msg)

        # Validate inputs
        if question is None:
            msg = "Either 'question' or 'full_content' must be provided"
            raise ValueError(msg)

        # Use question for similarity search
        query_question = question.strip()

        # Retrieve similar examples from training data
        # Retrieve more results if filtering for unique questions
        retrieval_k = self.icl_n * retrieve_k_multiplier if ensure_unique_questions else self.icl_n
        similar_docs = self.vector_store.similarity_search(query_question, k=retrieval_k)

        # Filter for unique questions if requested
        if ensure_unique_questions:
            seen_questions = set()
            unique_similar_docs = []

            for doc in similar_docs:
                metadata = doc.metadata
                messages = metadata.get("full_content", [])

                if not messages:
                    continue

                doc_content = messages[0]["content"]

                # Extract question from the doc content
                doc_question_extracted = ""
                if "Question:" in doc_content:
                    doc_question_extracted = doc_content.split("Question:")[1].split("\n")[0].strip()
                else:
                    doc_question_extracted = doc_content[:500]

                # Only add if we haven't seen this question before
                if doc_question_extracted not in seen_questions:
                    seen_questions.add(doc_question_extracted)
                    unique_similar_docs.append(doc)

                # Stop once we have enough unique examples
                if len(unique_similar_docs) >= self.icl_n:
                    break

            similar_docs = unique_similar_docs
            print(f"🔍 Filtered to {len(similar_docs)} unique question examples (retrieved {retrieval_k} total)")

        # Build ICL messages
        icl_messages = []

        for doc in similar_docs:
            metadata = doc.metadata
            messages = metadata.get("full_content", [])

            if not messages:
                continue

            doc_content = messages[0]["content"]
            if doc_type == "document":
                doc_question, doc_sentences, doc_indices = self._parse_document_ranking_content(doc_content)
            elif doc_type == "chunk":
                doc_question, doc_sentences, doc_indices = self._parse_chunk_ranking_content(doc_content)

            # Add assistant message with ground truth if available
            if include_ground_truth:
                example_qrel = metadata.get("qrel", {})

                if example_qrel:
                    # Extract top N most relevant sentence indices from qrel
                    # qrel maps: {sentence_index: relevance_score}
                    # We need to find the top N sentence_indices by score
                    scored_items = [(int(idx), score) for idx, score in example_qrel.items()]
                    scored_items.sort(key=lambda x: (-x[1], x[0]))  # Sort by score desc, then index asc

                    # Get top N indices (these are the actual sentence_indices, not list positions)
                    top_indices = [(idx, score) for idx, score in scored_items[: self.icl_n]]

                    selected_sentences = []
                    for idx, score in top_indices:
                        # Check if idx exists in doc_indices
                        if idx not in doc_indices:
                            print(f"⚠️ Warning: Index {idx} not found in doc_indices. Skipping this item.")
                            continue

                        # Get the position of idx in doc_indices
                        position = doc_indices.index(idx)

                        if doc_type == "document":
                            selected_sentences.append(
                                f"[Document Index {idx}] - Score {score}\nDocument: {doc_sentences[position]}"
                            )
                        elif doc_type == "chunk":
                            # Remove last 2 characters from chunk
                            chunk_content = (
                                doc_sentences[position][:-2]
                                if len(doc_sentences[position]) > 2
                                else doc_sentences[position]
                            )
                            selected_sentences.append(f"[Chunk Index {idx}] - Score {score}\nChunk: {chunk_content}")

                    # Format the response based on style
                    if format_style == "concise":
                        # Simple list format
                        response = f"Question: {doc_question}\n"
                        response += "\n".join(selected_sentences)
                    else:  # detailed
                        # Include reasoning
                        response = "Based on careful analysis of the question and available content, "
                        response += f"the most relevant items in order are: {top_indices}. "
                        response += f"Question: {doc_question}\n"
                        response += "\n".join(selected_sentences)

                    icl_messages.append({"role": "assistant", "content": response})

        print(f"✅ Retrieved {len(similar_docs)} ICL examples for question: '{query_question[:80]}...'")
        print(f"📝 Generated {len(icl_messages)} ICL messages ({len(icl_messages) // 2} pairs)")

        # Optional: Return metadata about the query for debugging
        if sentences:
            return icl_messages, {
                "query_question": query_question,
                "num_sentences": len(sentences),
                "sentence_indices": sentence_indices,
                "qrel": qrel,
            }
        return icl_messages

    def get_icl_for_chunk_ranking(self, full_content: dict, format_style: str = "concise") -> list[dict]:
        """Convenience wrapper for building ICL messages for chunk ranking.

        Args:
            full_content (Dict): Source content containing ``content`` and ``qrel``.
            format_style (str, optional): `"concise"` or `"detailed"`. Defaults to
                `"concise"`.

        Returns:
            List[Dict]: Assistant messages suitable for ICL.
        """
        result = self.get_icl_messages(
            full_content=full_content,
            format_style=format_style,
            doc_type="chunk",
        )

        # Handle tuple return (with metadata)
        if isinstance(result, tuple):
            return result[0]
        return result

    def get_icl_for_document_ranking(self, full_content: dict, format_style: str = "concise") -> list[dict]:
        """Convenience wrapper for building ICL messages for document ranking.

        Args:
            full_content (Dict): Source content containing ``content`` and ``qrel``.
            format_style (str, optional): `"concise"` or `"detailed"`. Defaults to
                `"concise"`.

        Returns:
            List[Dict]: Assistant messages suitable for ICL.
        """
        result = self.get_icl_messages(
            full_content=full_content,
            format_style=format_style,
            doc_type="document",
        )

        # Handle tuple return (with metadata)
        if isinstance(result, tuple):
            return result[0]
        return result

    def get_icl_for_financebench(
        self,
        samples_per_type: int = 3,
        random_seed: int | None = None,
    ) -> list[dict]:
        """Retrieve random ICL examples for FinanceBench dataset with balanced question types.

        Randomly samples examples from training data, ensuring equal representation
        across all question types found in the dataset.

        Args:
            samples_per_type (int, optional): Number of examples to retrieve per
                question_type. Defaults to 3.
            random_seed (int | None, optional): Random seed for reproducibility.
                If None, sampling will be non-deterministic. Defaults to None.

        Returns:
            List[Dict]: A list of user/assistant message pairs formatted as ICL examples.

        """
        self.training_data = self._load_training_data()
        if not self.training_data:
            msg = "Training data not loaded"
            raise ValueError(msg)

        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)

        # Group training data by question_type
        type_groups = {}
        for item in self.training_data:
            question_type = item.get("question_type", "")

            if question_type not in type_groups:
                type_groups[question_type] = []

            type_groups[question_type].append(item)
        # Randomly sample from each question type
        selected_items = []
        for question_type, items in type_groups.items():
            print(f"{question_type}: {len(items)} items")

            if samples_per_type <= 0:
                msg = "samples_per_type must be > 0"
                raise ValueError(msg)

            sample_size = min(samples_per_type, len(items))
            sampled = random.sample(items, sample_size)
            selected_items.extend(sampled)

        print(f"🔍 Sampled {len(selected_items)} examples across {len(type_groups)} question types:")
        for qtype, items in type_groups.items():
            print(f"  - {qtype}: {min(samples_per_type, len(items))} examples")

        # Build ICL messages
        icl_messages = []

        for item in selected_items:
            # Support both direct fields and messages format
            if "question" in item:
                # Direct format (top-level fields)
                doc_question = item.get("question", "")
                doc_answer = item.get("answer", "")
                doc_justification = item.get("justification", "")
                doc_question_type = item.get("question_type", "")
            else:
                # Messages format (nested in messages array)
                messages = item.get("messages", [])

                if not messages:
                    continue

                msg = messages[0]
                doc_question = msg.get("question", "")
                doc_answer = msg.get("answer", "")
                doc_justification = msg.get("justification", "")
                doc_question_type = msg.get("question_type", "")

            # Format user message
            user_content = f"Question: {doc_question}"
            if doc_question_type:
                user_content = f"[Type: {doc_question_type}]\n{user_content}"
                user_content += f"Answer: {doc_answer}"
                user_content += f"\n\nJustification: {doc_justification}"

            icl_messages.append({"role": "user", "content": user_content})

        print(f"✅ Generated {len(icl_messages)} ICL messages")

        return icl_messages

    def get_icl_for_fiqa(
        self,
        query_text: str,
        samples_per_retrieval: int = 5,
        format_style: str = "concise",
        ensure_unique_queries: bool = True,
    ) -> list[dict]:
        """Retrieve ICL examples for FiQA dataset using similarity search.

        Performs vector similarity search to find relevant examples from the training
        data based on the input query. Returns formatted user/assistant message pairs
        showing questions and their relevant documents.

        Args:
            query_text (str): The financial question to find similar examples for.
            samples_per_retrieval (int, optional): Number of ICL examples to retrieve.
                Defaults to 5.
            format_style (str, optional): Output style, either "concise" or "detailed".
                Defaults to "concise".
            ensure_unique_queries (bool, optional): If True, filters results to ensure
                unique question texts. Defaults to True.
            retrieve_k_multiplier (int, optional): Multiplier for initial retrieval
                when ensuring unique queries. Defaults to 3.

        Returns:
            List[Dict]: A list of user/assistant message pairs formatted as ICL examples.
                Each pair shows a question and its relevant documents with relevance scores.
        """
        if not self.vector_store:
            msg = "Vector store not initialized"
            raise ValueError(msg)

        # Validate input
        if not query_text or not query_text.strip():
            msg = "query_text cannot be empty"
            raise ValueError(msg)

        query_text = query_text.strip()

        # Retrieve similar examples from training data
        similar_docs = self.vector_store.similarity_search(query_text, k=samples_per_retrieval)

        # Filter for unique queries if requested
        if ensure_unique_queries:
            seen_queries = set()
            unique_similar_docs = []

            for doc in similar_docs:
                # The page_content contains the query text
                doc_query = doc.page_content.strip()

                # Only add if we haven't seen this query before
                if doc_query not in seen_queries:
                    seen_queries.add(doc_query)
                    unique_similar_docs.append(doc)

                # Stop once we have enough unique examples
                if len(unique_similar_docs) >= samples_per_retrieval:
                    break

            similar_docs = unique_similar_docs
            print(f"🔍 Filtered to {len(similar_docs)} unique query examples (retrieved {samples_per_retrieval} total)")

        # Build ICL messages
        icl_messages = []

        for doc in similar_docs:
            metadata = doc.metadata
            doc_query = doc.page_content

            # Get relevant documents from metadata
            relevant_docs = metadata.get("relevant_docs", [])
            relevance_scores = metadata.get("relevance_scores", {})

            if not relevant_docs:
                # Skip if no relevant documents
                continue

            # Format user message (the question)
            user_content = f"Question: {doc_query}"
            icl_messages.append({"role": "user", "content": user_content})

            # Format assistant message (the relevant documents with scores)
            if format_style == "concise":
                # Simple list format with relevance scores
                response_parts = []
                for doc_id in relevant_docs:
                    score = relevance_scores.get(doc_id, 1)  # Default to 1 if not specified
                    response_parts.append(f"Document ID: {doc_id} (Relevance: {score})")

                response = "Relevant documents:\n" + "\n".join(response_parts)
            else:  # detailed
                # Include reasoning
                response = f"Based on the question '{doc_query}', "
                response += f"I identified {len(relevant_docs)} relevant documents. "
                response += "The most relevant documents are:\n"

                for doc_id in relevant_docs:
                    score = relevance_scores.get(doc_id, 1)
                    response += f"- Document ID: {doc_id} (Relevance Score: {score})\n"

            icl_messages.append({"role": "assistant", "content": response})

        print(f"✅ Retrieved {len(similar_docs)} ICL examples for query: '{query_text[:80]}...'")

        return icl_messages
