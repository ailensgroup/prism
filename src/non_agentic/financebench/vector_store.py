import os
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.non_agentic.financebench.configs import (
    DATASET_PORTION,
    PATH_DATASET_JSONL,
    PATH_DOCUMENT_INFO_JSONL,
    PATH_PDFS,
    VS_CHUNK_OVERLAP,
    VS_CHUNK_SIZE,
    VS_DIR_VS,
)

load_dotenv()

# Load Full Dataset
df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
df_meta = pd.read_json(PATH_DOCUMENT_INFO_JSONL, lines=True)
df_full = df_questions.merge(df_meta, on="doc_name")

df_questions = df_questions.sort_values("doc_name")
ALL_DOCS = df_questions["doc_name"].unique().tolist()
print(f"Total number of distinct PDF: {len(ALL_DOCS)}")

if DATASET_PORTION != "ALL":
    df_questions = df_questions.loc[df_questions["dataset_subset_label"] == DATASET_PORTION]
print(f"Number of questions: {len(df_questions)}")

df_questions = df_questions.sort_values("doc_name")
docs = df_questions["doc_name"].unique().tolist()
print(f"Number of distinct PDF: {len(docs)}")


# HELPER FUNCTIONS (PDF-PARSING + VECTOR-STORE SETUPS)
def get_pdf_text(doc: str) -> PyMuPDFLoader:
    """Get PDF Text from documents."""
    path_doc = f"{PATH_PDFS}/{doc}.pdf"
    pdf_reader = PyMuPDFLoader(path_doc)
    return pdf_reader.load()


def build_vectorstore_retriever(
    docs: str, embeddings: AzureOpenAIEmbeddings, db_path: str = VS_DIR_VS
) -> tuple[VectorStoreRetriever, Chroma]:
    """Build or load a Chroma vector store retriever.

    Args:
        docs: Document name or "all" for all documents
        embeddings: Embedding function (OpenAIEmbeddings or AzureOpenAIEmbeddings)
        db_path: Vector store parent path

    Returns:
        tuple: (retriever, vectordb)
    """
    if docs == "all":
        docs = ALL_DOCS
        db_path = db_path + "/shared"
    else:
        if isinstance(docs, str):
            docs = [docs]
        db_path = db_path + "/" + docs[0]

    if not os.path.exists(db_path):
        print(f"📁 Creating new vector store at: {db_path}")
        os.makedirs(db_path, exist_ok=True)
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        for doc in docs:
            print(f"📄 Processing document: {doc}")
            pdf_text = get_pdf_text(doc)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=VS_CHUNK_SIZE,
                chunk_overlap=VS_CHUNK_OVERLAP,
            )
            splitted_texts = text_splitter.split_documents(pdf_text)

            print(f"   Added {len(splitted_texts)} chunks to vector store")

            vectordb.add_documents(documents=splitted_texts)

        print(f"Vector store created with {len(docs)} document(s)")

    else:
        print(f"Loading existing vector store from: {db_path}")
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    return vectordb.as_retriever(), vectordb


def build_vectorstore_retriever_fiqa(
    embeddings: AzureOpenAIEmbeddings,
    db_path: str = VS_DIR_VS,
    docs_dict: dict | None = None,
    search_kwargs: dict | None = None,
    batch_size: int = 1000,
    delay_between_batches: float = 2.0,
) -> tuple[VectorStoreRetriever, Chroma]:
    """Build or load a Chroma vector store retriever with batching."""
    if docs_dict is None:
        msg = "docs_dict is required for FiQA dataset"
        raise ValueError(msg)
    docs = list(docs_dict.keys())
    db_path = db_path + "/fiqa_shared"

    if not os.path.exists(db_path):
        print(f"Creating new vector store at: {db_path}")
        os.makedirs(db_path, exist_ok=True)

        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

        if docs_dict is None:
            msg = "docs_dict is required for FiQA dataset"
            raise ValueError(msg)

        print(f"Preparing {len(docs)} FiQA text documents...")
        all_documents = []

        for doc_id in tqdm(docs, desc="Preparing documents"):
            if doc_id not in docs_dict:
                print(f"⚠️ Warning: Document {doc_id} not found in docs_dict")
                continue

            text = docs_dict[doc_id]

            doc = Document(
                page_content=text,
                metadata={
                    "doc_id": str(doc_id),
                    "source": "fiqa",
                    "type": "financial_qa",
                },
            )
            all_documents.append(doc)

        print(f"Prepared {len(all_documents)} documents")
        avg_length = sum(len(doc.page_content) for doc in all_documents) / len(all_documents) if all_documents else 0
        if avg_length > VS_CHUNK_SIZE:
            print(f"   Average text length ({avg_length:.0f} chars) exceeds chunk size, splitting...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=VS_CHUNK_SIZE,
                chunk_overlap=VS_CHUNK_OVERLAP,
            )
            all_documents = text_splitter.split_documents(all_documents)
            for chunk in all_documents:
                if "doc_id" in chunk.metadata:
                    chunk.metadata["doc_id"] = str(chunk.metadata["doc_id"])
            print(f"   Split into {len(all_documents)} chunks")

        print(f"\nAdding documents in batches of {batch_size}...")
        total_batches = (len(all_documents) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding batches", total=total_batches):
            batch = all_documents[i : i + batch_size]

            try:
                vectordb.add_documents(documents=batch)
                print(f"   ✅ Added batch {i // batch_size + 1}/{total_batches} ({len(batch)} documents)")
                if i + batch_size < len(all_documents):
                    print(f"   ⏳ Waiting {delay_between_batches}s before next batch...")
                    time.sleep(delay_between_batches)

            except Exception as e:
                print(f"Error adding batch {i // batch_size + 1}: {e}")
                if "429" in str(e) or "RateLimitError" in str(e):
                    print("   ⏳ Rate limit hit, waiting 10 seconds...")
                    time.sleep(10)

                    try:
                        vectordb.add_documents(documents=batch)
                        print(f"Retry successful for batch {i // batch_size + 1}")
                    except Exception as e2:
                        print(f"Retry failed: {e2}")
                        print(f"Skipping batch {i // batch_size + 1}")
                        continue
                else:
                    raise

        print(f"\nVector store created with {len(docs)} document(s)")
        print("\nVerifying document metadata...")
        try:
            test_results = vectordb.similarity_search("test", k=3)
            if test_results:
                for i, doc in enumerate(test_results):
                    doc_id = doc.metadata.get("doc_id", "MISSING")
                    print(f"   Sample {i + 1}: doc_id={doc_id}")
        except Exception as e:
            print(f"   Could not verify metadata: {e}")

    else:
        print(f"Loading existing vector store from: {db_path}")
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

        try:
            collection_count = len(vectordb.get()["ids"])
            print(f"   Found {collection_count} documents in vector store")

            test_results = vectordb.similarity_search("test", k=3)
            if test_results:
                print("   Sample metadata check:")
                for i, doc in enumerate(test_results):
                    doc_id = doc.metadata.get("doc_id", "MISSING")
                    print(f"      Doc {i + 1}: doc_id={doc_id}")
        except Exception as e:
            print(f"   Could not verify document count: {e}")

    if search_kwargs is None:
        search_kwargs = {"k": 500}

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    print("\nTesting retriever...")
    try:
        test_docs = retriever.invoke("test financial question")
        print(f"Retrieved {len(test_docs)} test documents")
        if test_docs:
            sample_doc = test_docs[0]
            doc_id = sample_doc.metadata.get("doc_id", "MISSING")
            print(f"   Sample doc_id: {doc_id}")
    except Exception as e:
        print(f"Retriever test failed: {e}")

    return retriever, vectordb


if __name__ == "__main__":
    retriever, vectordb = build_vectorstore_retriever(docs="all")
