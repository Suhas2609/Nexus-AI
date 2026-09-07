import argparse
from pathlib import Path

from rag.vectorstore.store import get_or_create_store
from rag.vectorstore.retriever import get_mmr_retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Developer utility for direct vector store queries via MMR retrieval."
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Query string to evaluate against the vector store.",
    )
    args = parser.parse_args()

    store = get_or_create_store()
    retriever = get_mmr_retriever(store)

    docs = retriever.invoke(args.question)

    print(f"\n[INFO] Query: {args.question}")
    print(f"[INFO] Retrieved {len(docs)} document chunks:\n")

    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "N/A")
        content_preview = doc.page_content[:500].strip()

        print(f"--- Result {i} ---")
        print(f"Source: {source} | Page: {page}")
        print(f"Content Preview:\n{content_preview}\n")


if __name__ == "__main__":
    main()