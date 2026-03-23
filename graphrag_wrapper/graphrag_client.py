"""Microsoft GraphRAG client wrapper.

Provides a high-level interface for GraphRAG querying and indexing,
with fallback handling when GraphRAG is not yet initialized.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class GraphRAGClient:
    """Wrapper around Microsoft GraphRAG for multi-hop legal reasoning.

    Manages GraphRAG lifecycle: setup verification, document indexing,
    and query execution. Falls back gracefully when not available.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize GraphRAG client.

        Args:
            config: Full config dict. If None, loads from configs/config.yaml.
        """
        if config is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        graphrag_config = config["graphrag"]
        self.working_dir = Path(graphrag_config["working_dir"]).resolve()
        self.query_type: str = graphrag_config.get("query_type", "local")
        self.local_path = Path(graphrag_config["local_path"]).resolve()

        self._indexed: bool = False
        self._check_indexed()

        logger.info(
            "GraphRAGClient initialized | working_dir={} | query_type={} | indexed={}",
            self.working_dir,
            self.query_type,
            self._indexed,
        )

    def _check_indexed(self) -> None:
        """Check if GraphRAG has completed indexing."""
        output_dir = self.working_dir / "output"
        if output_dir.exists():
            # Check for parquet or json output files indicating completed indexing
            parquet_files = list(output_dir.rglob("*.parquet"))
            json_files = list(output_dir.rglob("*.json"))
            self._indexed = len(parquet_files) > 0 or len(json_files) > 0
        else:
            self._indexed = False

    def is_available(self) -> bool:
        """Check if GraphRAG is set up and has indexed data.

        Returns:
            True if GraphRAG is ready for queries.
        """
        self._check_indexed()
        return self._indexed

    def answer(self, query: str, history: str | None = None) -> str:
        """Perform multi-hop legal reasoning via GraphRAG.

        Args:
            query: User question requiring multi-hop reasoning.
            history: Optional conversation history string.

        Returns:
            Answer string from GraphRAG.

        Raises:
            RuntimeError: If GraphRAG is not available or query fails.
        """
        if not self.is_available():
            logger.warning("GraphRAG not indexed, cannot answer query")
            raise RuntimeError(
                "GraphRAG chưa được index. Chạy index_documents() trước."
            )

        # Build query with history context
        full_query = query
        if history:
            full_query = f"Lịch sử hội thoại:\n{history}\n\nCâu hỏi: {query}"

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "graphrag.query",
                    "--root", str(self.working_dir),
                    "--method", self.query_type,
                    "--query", full_query,
                ],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(self.working_dir),
            )
            if result.returncode != 0:
                logger.error("GraphRAG query failed: {}", result.stderr[:500])
                raise RuntimeError(f"GraphRAG query error: {result.stderr[:200]}")

            answer = result.stdout.strip()
            logger.info("GraphRAG answered | length={}", len(answer))
            return answer

        except subprocess.TimeoutExpired:
            logger.error("GraphRAG query timed out")
            raise RuntimeError("GraphRAG query timed out after 180s")

    def index_documents(self, doc_dir: str | Path) -> None:
        """Trigger GraphRAG indexing on a directory of documents.

        Copies documents to the GraphRAG input directory and runs
        the indexing pipeline.

        Args:
            doc_dir: Directory containing text documents to index.

        Raises:
            RuntimeError: If indexing fails.
        """
        doc_path = Path(doc_dir)
        if not doc_path.exists():
            raise RuntimeError(f"Document directory not found: {doc_path}")

        # Copy documents to GraphRAG input
        input_dir = self.working_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        file_count = 0
        for src_file in doc_path.iterdir():
            if src_file.suffix in {".txt", ".json", ".md"}:
                dst = input_dir / src_file.name
                shutil.copy2(src_file, dst)
                file_count += 1

        logger.info("Copied {} files to GraphRAG input at {}", file_count, input_dir)

        if file_count == 0:
            logger.warning("No documents to index")
            return

        # Run GraphRAG indexing
        logger.info("Starting GraphRAG indexing...")
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "graphrag.index",
                    "--root", str(self.working_dir),
                ],
                capture_output=True,
                text=True,
                timeout=3600,  # Indexing can take a long time
                cwd=str(self.working_dir),
            )
            if result.returncode != 0:
                logger.error("GraphRAG indexing failed: {}", result.stderr[:500])
                raise RuntimeError(f"GraphRAG indexing error: {result.stderr[:300]}")

            self._check_indexed()
            logger.info("GraphRAG indexing complete | indexed={}", self._indexed)

        except subprocess.TimeoutExpired:
            logger.error("GraphRAG indexing timed out after 1 hour")
            raise RuntimeError("GraphRAG indexing timed out")
