"""TopicNodeParser subclass: per-paragraph progress, library print capture, skip accounting."""
import contextlib
import io
from typing import Any, List, Optional, Sequence
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, Document
from llama_index.node_parser.topic import TopicNodeParser
from src.logger import OllenLogger

log = OllenLogger("chunker")


class OllenTopicNodeParser(TopicNodeParser):
    """TopicNodeParser with observability.

    - progress_cb (assignable after from_defaults) receives processed/total paragraph
      fractions (0-1); the ingestion layer maps them into the chunking percent band.
    - The library prints raw noise ('No valid JSON found...') and silently returns []
      when the LLM answers without JSON: stdout is captured per call (no logic copied),
      skipped fragments are DEBUG-logged with a preview and totalled at INFO.
    """
    progress_cb: Optional[Any] = Field(default=None, exclude=True)
    _total_paragraphs: int = PrivateAttr(default=0)
    _processed_paragraphs: int = PrivateAttr(default=0)
    _skipped_fragments: int = PrivateAttr(default=0)

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Record the paragraph total (progress denominator) and reset counters."""
        paragraphs = super().split_into_paragraphs(text)
        self._total_paragraphs = len(paragraphs)
        self._processed_paragraphs = 0
        self._skipped_fragments = 0
        log.info("topic chunking: %d paragraph(s)", len(paragraphs))
        return paragraphs

    def proposition_transfer(self, paragraph: str) -> List[str]:
        """One LLM proposition call per paragraph: capture print noise, count, report progress."""
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            propositions = super().proposition_transfer(paragraph)
        if not propositions:
            # Library returns [] (fragment dropped) when the LLM reply has no JSON
            self._skipped_fragments += 1
            log.debug("fragment skipped (no propositions): %.120s", paragraph)
        noise = captured.getvalue().strip()
        if noise:
            log.debug("library notice: %.200s", noise)
        self._processed_paragraphs += 1
        if self.progress_cb is not None and self._total_paragraphs:
            self.progress_cb(self._processed_paragraphs / self._total_paragraphs)
        return propositions

    def build_topic_based_nodes_from_documents(self, documents: Sequence[Document]) -> List[BaseNode]:
        """Delegate to the library, then report how many fragments were dropped."""
        nodes = super().build_topic_based_nodes_from_documents(documents)
        if self._skipped_fragments:
            log.info("topic chunking: %d fragment(s) skipped (LLM returned no propositions)", self._skipped_fragments)
        return nodes
