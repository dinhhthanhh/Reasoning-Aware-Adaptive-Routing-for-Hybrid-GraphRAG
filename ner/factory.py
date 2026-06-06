"""NER model factory.

Selects the appropriate NER model (ViNER or EnNER) based on configuration.
"""

from typing import Any

from loguru import logger

from ner.en_ner import EnNER
from ner.vi_ner import ViNER


_model_cache: dict[str, Any] = {}

def get_ner_model(config: dict[str, Any] | None = None) -> ViNER | EnNER:
    """Instantiate the correct NER model based on the model name in config.

    Args:
        config: The `ner` section of the config dictionary.

    Returns:
        An instance of ViNER or EnNER.
    """
    model_name = ""
    if config:
        model_name = config.get("model_name", "")

    if model_name in _model_cache:
        logger.debug("Factory returning cached NER model: {}", model_name)
        return _model_cache[model_name]

    # If the model name suggests an English generic model, use EnNER
    if "bert-base-ner" in model_name.lower() or "english" in model_name.lower():
        logger.info("Factory selected EnNER based on model_name: {}", model_name)
        model = EnNER(config)
    else:
        # Default to ViNER
        logger.info("Factory selected ViNER based on model_name: {}", model_name)
        model = ViNER(config)
        
    _model_cache[model_name] = model
    return model
