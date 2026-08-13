import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.ingestion.pipeline import ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Nexus AI vector store.")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    print(ingest(args.path))


if __name__ == "__main__":
    main()
