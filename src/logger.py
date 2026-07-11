"""Class-based logging facade: one configured 'ollen_rag' hierarchy, per-module instances."""
import logging
import sys
from src.settings import Settings, get_settings

# Names accepted for OLLEN_RAG_LOG_LEVEL (case-insensitive); anything else -> INFO
_VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class OllenLogger:
    """Thin facade over stdlib logging.

    OllenLogger.setup(settings) configures the shared 'ollen_rag' parent logger once
    (stdout handler, level from settings.log_level, no propagation to uvicorn's root);
    OllenLogger("module") instances delegate debug/info/warning/error to the
    'ollen_rag.<module>' child, which inherits the parent's handler and level.
    """
    PARENT = "ollen_rag"
    # The one stdout handler this class owns; identity-tracked so setup stays idempotent
    # even when other tooling (e.g. pytest's caplog) attaches its own handlers
    _handler: logging.Handler | None = None

    def __init__(self, name: str):
        # Child logger: inherits parent handler/level, keeps the module name in output
        self._logger = logging.getLogger(f"{self.PARENT}.{name}")

    @classmethod
    def setup(cls, settings: Settings | None = None) -> None:
        """Configure the parent logger. Idempotent: never stacks a second handler."""
        settings = settings or get_settings()
        parent = logging.getLogger(cls.PARENT)
        level_name = settings.log_level.upper()
        valid = level_name in _VALID_LEVELS
        parent.setLevel(getattr(logging, level_name) if valid else logging.INFO)
        # Don't bubble to the root logger — uvicorn configures it and would double-print
        parent.propagate = False
        if cls._handler is None or cls._handler not in parent.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s — %(message)s"))
            parent.addHandler(handler)
            cls._handler = handler
        if not valid:
            parent.warning("Invalid log level '%s', falling back to INFO", settings.log_level)

    def debug(self, msg: str, *args) -> None:
        """DEBUG-level message (printf-style args, like stdlib logging)."""
        self._logger.debug(msg, *args)

    def info(self, msg: str, *args) -> None:
        """INFO-level message."""
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        """WARNING-level message."""
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        """ERROR-level message."""
        self._logger.error(msg, *args)
