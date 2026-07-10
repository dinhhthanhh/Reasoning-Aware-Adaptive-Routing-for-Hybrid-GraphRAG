"""Auto-setup for Microsoft GraphRAG.

Handles cloning the GraphRAG repository, installing its dependencies,
and initializing the workspace for local usage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def setup_graphrag(config: dict[str, Any] | None = None) -> None:
    """Clone Microsoft GraphRAG if not present and install dependencies.

    Does not modify GraphRAG source code. Sets up the environment
    for local inference with Ollama.

    Args:
        config: GraphRAG config dict. If None, loads from configs/config.yaml.

    Raises:
        RuntimeError: If git clone or pip install fails.
    """
    if config is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        config = full_config["graphrag"]

    repo_url: str = config["repo_url"]
    local_path = Path(config["local_path"]).resolve()
    working_dir = Path(config["working_dir"]).resolve()

    # Clone if not present
    if not local_path.exists():
        logger.info("Cloning Microsoft GraphRAG from {}", repo_url)
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(local_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("GraphRAG cloned to {}", local_path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to clone GraphRAG: {exc.stderr}") from exc
    else:
        logger.info("GraphRAG already cloned at {}", local_path)

    # Install GraphRAG as a package
    logger.info("Installing GraphRAG dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(local_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("GraphRAG dependencies installed")
    except subprocess.CalledProcessError as exc:
        logger.warning("GraphRAG pip install warning: {}", exc.stderr[:500])
        # Try installing just from pip as fallback
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "graphrag"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("GraphRAG installed from PyPI as fallback")
        except subprocess.CalledProcessError as exc2:
            raise RuntimeError(f"Failed to install GraphRAG: {exc2.stderr}") from exc2

    # Create working directory
    working_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GraphRAG working directory: {}", working_dir)

    # Initialize GraphRAG workspace if settings.yaml doesn't exist
    settings_file = working_dir / "settings.yaml"
    if not settings_file.exists():
        _create_settings(working_dir, config)
        logger.info("GraphRAG settings created at {}", settings_file)


def _create_settings(working_dir: Path, config: dict[str, Any]) -> None:
    """Create GraphRAG settings.yaml for local Ollama usage.

    Args:
        working_dir: GraphRAG workspace directory.
        config: GraphRAG config dict.
    """
    # Load ollama config for model settings
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    ollama_config = full_config["ollama"]

    settings = {
        "llm": {
            "api_key": "ollama",
            "type": "openai_chat",
            "model": ollama_config["model"],
            "api_base": f"{ollama_config['base_url']}/v1",
            "max_tokens": ollama_config.get("max_tokens", 1024),
            "temperature": ollama_config.get("temperature", 0.1),
        },
        "embeddings": {
            "llm": {
                "api_key": "ollama",
                "type": "openai_embedding",
                "model": f"{ollama_config['base_url']}/v1",
            },
        },
        "input": {
            "type": "file",
            "file_type": "text",
            "base_dir": "input",
        },
        "chunks": {
            "size": 512,
            "overlap": 64,
        },
        "cache": {
            "type": "file",
            "base_dir": "cache",
        },
        "reporting": {
            "type": "file",
            "base_dir": "output",
        },
        "storage": {
            "type": "file",
            "base_dir": "output",
        },
    }

    settings_path = working_dir / "settings.yaml"
    with open(settings_path, "w", encoding="utf-8") as f:
        yaml.dump(settings, f, default_flow_style=False, allow_unicode=True)

    # Create input directory
    (working_dir / "input").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_graphrag()
