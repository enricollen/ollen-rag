"""watsonx.ai LLM connector, self-registered with the LLM factory on import."""
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from src.factories.llm import LLMConnector, LLMConnectorFactory
from src.settings import Settings, get_settings

@LLMConnectorFactory.register("watsonx")
class WatsonxConnector(LLMConnector):
    """LLM connector backed by the ibm-watsonx-ai SDK; lazily connects on first use.

    Uses the chat completions endpoint (ModelInference.chat(), /ml/v1/text/chat) rather than
    the raw text-generation endpoint. The raw endpoint has no chat template, so an instruct
    model (e.g. llama-3-3-70b-instruct) has no learned stop token there and drifts into fake
    new turns or paraphrase-repeats past the real answer until max_new_tokens forcibly (and
    mid-sentence) truncates it. The chat endpoint applies the model's actual chat template
    and stop tokens, so generation ends on its own once the answer is complete.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: ModelInference | None = None
        # Metadata create_llm() reads to build the llamaindex adapter, keeping it settings-blind.
        self.model_name = self._settings.watsonx_llm_model_id
        self.max_new_tokens = self._settings.watsonx_max_new_tokens

    def _get_model(self) -> ModelInference:
        """Create the SDK client on first use so constructing the connector needs no network."""
        if self._model is None:
            self._model = ModelInference(
                model_id=self._settings.watsonx_llm_model_id,
                credentials=Credentials(url=self._settings.watsonx_url, api_key=self._settings.watsonx_apikey),
                project_id=self._settings.watsonx_project_id,
            )
        return self._model

    def complete(self, prompt: str) -> str:
        """Send *prompt* as a single user chat turn and return the assistant text."""
        params = {
            "max_tokens": self._settings.watsonx_max_new_tokens,
            "temperature": self._settings.watsonx_temperature,
            "repetition_penalty": self._settings.watsonx_repetition_penalty,
        }
        response = self._get_model().chat(messages=[{"role": "user", "content": prompt}], params=params)
        return response["choices"][0]["message"]["content"]