"""Shared litellm bootstrap: imported by every litellm-backed connector before the library loads,
so litellm's global flags are set exactly once, in one place.

litellm prints a red "Provider List: https://docs.litellm.ai/docs/providers" banner straight to
stdout every time its provider-resolution logic meets a model string it can't map. Its cost/logging
layer probes that best-effort after each successful call and swallows the resulting error — but the
print has already fired, so a normal eval emits dozens of these lines. `suppress_debug_info` is
litellm's own gate on that print (see litellm_core_utils/get_llm_provider_logic.py); turning it on
keeps our logs clean without hiding real errors, which still surface as raised exceptions."""
import logging
import os

# Must precede the first litellm import: litellm.__init__ runs dotenv.load_dotenv() under its default
# DEV mode, which would splice this project's whole .env into os.environ. Settings owns .env loading.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import litellm  # noqa: E402

# Silence the per-call "Provider List" banner and litellm's verbose logging; keep its logger at
# WARNING so genuine failures are still visible.
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
