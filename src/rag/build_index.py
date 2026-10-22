"""CLI script to build or incrementally update the RAG knowledge index."""

from __future__ import annotations

import argparse
import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.indexer import Indexer
from src.rag.config import load_rag_config


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description="Build or update the RAG knowledge base index.")
    parser.add_argument("--rebuild", action="store_true", help="Force a complete rebuild from scratch.")
    parser.add_argument("--update", action="store_true", help="Incrementally update changed files (default).")
    args = parser.parse_args()

    config = load_rag_config()
    indexer = Indexer(config)

    t0 = time.time()
    if args.rebuild:
        print("[INFO] Initiating complete index rebuild...")
        result = indexer.rebuild()
    else:
        print("[INFO] Checking for modified files and performing incremental update...")
        result = indexer.update_index()

    elapsed = time.time() - t0
    print(f"[SUCCESS] Index operation: {result['status']}")
    print(f"  - Knowledge Version: {result.get('knowledge_version')}")
    print(f"  - Indexed Files:     {result.get('files_count')}")
    print(f"  - Total Chunks:      {result.get('chunks_count')}")
    print(f"  - Elapsed Time:      {elapsed:.2f}s")
    if "added" in result and result["added"]:
        print(f"  - Added: {len(result['added'])} files (details in project_index_meta.json)")
    if "modified" in result and result["modified"]:
        print(f"  - Modified: {len(result['modified'])} files")
    if "removed" in result and result["removed"]:
        print(f"  - Removed ({len(result['removed'])}): {result['removed']}")


if __name__ == "__main__":
    main()
