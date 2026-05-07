from langchain_core.embeddings import Embeddings
from openai import OpenAI


class AzureCohereEmbeddings(Embeddings):
    def __init__(self, endpoint: str, api_key: str, deployment: str) -> None:
        """Initializes the AzureCohereEmbeddings instance."""
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        self.deployment = deployment

    def _embed(self, texts: list[str], input_type: str) -> list:
        """Generates embeddings for a list of documents using the Azure Cohere API."""
        batch_size = 16
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment,
                extra_body={"input_type": input_type},  # Cohere-specific param
            )
            results.extend([x.embedding for x in response.data])
        return results

    def embed_documents(self, texts: list[str]) -> list:
        """Generates embeddings for a list of documents using the Azure Cohere API."""
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        """Generates an embedding for a single query using the Azure Cohere API."""
        return self._embed([text], input_type="query")[0]
