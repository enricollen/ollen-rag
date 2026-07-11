"""LLM keyword enrichment: an IngestionPipeline transform that stamps search keywords on chunks."""
import re
from typing import Any
from llama_index.core.schema import BaseNode, TransformComponent
from src.logger import OllenLogger

log = OllenLogger("enrichment")

# Hard cap on keywords kept per chunk — beyond this the LLM is padding, not helping recall
MAX_KEYWORDS = 15
# Leading list decorations LLMs love: "1.", "2)", "-", "*", "•"
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*")


def parse_keywords(raw: str) -> str:
    """Defensively normalize an LLM keyword response into a comma-joined string.

    Splits on newlines and commas, strips list numbering/bullets and surrounding
    quotes, drops empties and duplicates (order-preserving), caps at MAX_KEYWORDS.
    Returns "" when nothing survives.
    """
    keywords: list[str] = []
    for part in re.split(r"[\n,]", raw):
        cleaned = _LIST_PREFIX.sub("", part).strip().strip("\"'").strip()
        if cleaned and cleaned not in keywords:
            keywords.append(cleaned)
    return ", ".join(keywords[:MAX_KEYWORDS])


class KeywordEnricher(TransformComponent):
    """Per-node LLM keyword extraction stored as metadata['keywords'] (comma-joined string).

    Keywords stay visible to the embedding (MetadataMode.EMBED includes metadata —
    they enrich the dense leg for free) but are excluded from the generation LLM
    context. LLM errors propagate: enrichment is opt-in, so a clear job failure
    beats silent partial enrichment.
    """
    llm: Any
    prompt: Any
    # Optional fraction callback (0-1 per node); the ingestion layer maps it into a percent band
    progress_cb: Any = None

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Enrich each node in place with LLM-extracted keywords; return the same list."""
        total = len(nodes)
        empty = 0
        for i, node in enumerate(nodes, start=1):
            raw = self.llm.predict(self.prompt, chunk_text=node.get_content())
            keywords = parse_keywords(raw)
            if keywords:
                node.metadata["keywords"] = keywords
                if "keywords" not in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.append("keywords")
                log.debug("chunk %d/%d keywords: %s", i, total, keywords)
            else:
                empty += 1
            if self.progress_cb is not None and total:
                self.progress_cb(i / total)
        log.info("keyword enrichment: %d chunk(s), %d without keywords", total, empty)
        return nodes
