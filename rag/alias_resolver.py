import re
from loguru import logger

class AliasResolver:
    """Resolves colloquial legal aliases into canonical document IDs.
    
    This helps the NER model and Graph database retrieve the exact node
    even if the user uses a shorthand like "NĐ 15" instead of "15/2020/NĐ-CP".
    """
    
    # Common legal aliases mapped to their canonical law_id.
    # In a production system, this would be loaded from a database or a large JSON file.
    DEFAULT_ALIASES = {
        r"\b(?:nđ|nghị định)\s*15\b": "15/2020/NĐ-CP",
        r"\b(?:nđ|nghị định)\s*100\b": "100/2019/NĐ-CP",
        r"\b(?:luật)\s*đất đai\s*(?:năm)?\s*2024\b": "31/2024/QH15",
        r"\b(?:luật)\s*đất đai\s*(?:năm)?\s*2013\b": "45/2013/QH13",
        r"\b(?:blhs|bộ luật hình sự)\s*(?:năm)?\s*2015\b": "100/2015/QH13",
        r"\b(?:bộ luật dân sự|blds)\s*(?:năm)?\s*2015\b": "91/2015/QH13",
        r"\b(?:luật)\s*hôn nhân\s*(?:và)?\s*gia đình\s*(?:năm)?\s*2014\b": "52/2014/QH13",
        r"\b(?:luật)\s*doanh nghiệp\s*(?:năm)?\s*2020\b": "59/2020/QH14",
    }

    def __init__(self, aliases: dict[str, str] = None) -> None:
        """Initialize with an optional custom dictionary of aliases.
        
        The keys should be regex patterns (case-insensitive) and the values 
        should be the canonical replacement string.
        """
        self.aliases = aliases if aliases is not None else self.DEFAULT_ALIASES
        # Precompile regexes for performance
        self._compiled_aliases = [
            (re.compile(pattern, re.IGNORECASE), canonical) 
            for pattern, canonical in self.aliases.items()
        ]
        
    def resolve(self, text: str) -> str:
        """Replace all aliases in the text with their canonical forms."""
        if not text:
            return text
            
        resolved_text = text
        for pattern, canonical in self._compiled_aliases:
            resolved_text = pattern.sub(f" {canonical} ", resolved_text)
            
        # Clean up any extra spaces introduced by substitution
        resolved_text = re.sub(r'\s+', ' ', resolved_text).strip()
        
        if resolved_text != text:
            logger.debug(f"AliasResolver: Normalized '{text}' -> '{resolved_text}'")
            
        return resolved_text
