"""Tests for the OllenLogger facade (handler idempotence, levels, delegation)."""
import logging
import pytest
from src.logger import OllenLogger
from src.settings import Settings


@pytest.fixture(autouse=True)
def _reset_parent_logger():
    # Global logging state must not leak between tests: restore handlers/level/propagate
    # and the class's own-handler marker (pytest's caplog adds foreign handlers too)
    parent = logging.getLogger("ollen_rag")
    saved = parent.handlers[:]
    saved_own = OllenLogger._handler
    parent.handlers.clear()
    OllenLogger._handler = None
    yield
    parent.handlers[:] = saved
    OllenLogger._handler = saved_own
    parent.setLevel(logging.NOTSET)
    parent.propagate = True


def test_setup_is_idempotent():
    # Idempotence = our own handler attached exactly once (foreign handlers may coexist)
    OllenLogger.setup(Settings(_env_file=None))
    first = OllenLogger._handler
    OllenLogger.setup(Settings(_env_file=None))
    assert OllenLogger._handler is first
    assert logging.getLogger("ollen_rag").handlers.count(first) == 1


def test_setup_honors_level_case_insensitive():
    OllenLogger.setup(Settings(_env_file=None, log_level="debug"))
    assert logging.getLogger("ollen_rag").level == logging.DEBUG


def test_setup_invalid_level_falls_back_to_info():
    OllenLogger.setup(Settings(_env_file=None, log_level="banana"))
    assert logging.getLogger("ollen_rag").level == logging.INFO


def test_instance_delegates_to_child_logger(caplog):
    # No setup() here: parent must keep default propagate=True so caplog sees records
    with caplog.at_level(logging.INFO, logger="ollen_rag.testmod"):
        OllenLogger("testmod").info("ciao %s", "mondo")
    assert "ciao mondo" in caplog.text
    assert caplog.records[0].name == "ollen_rag.testmod"
