import logging
from unittest import TestCase
from unittest.mock import MagicMock

from language_model_service_api.languagemodelservice_api import (
    TokenUsage,
)
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    GenericChatCompletionResponse,
)
from palantir_models.models import (
    GenericChatCompletionLanguageModel,
)

from langchain_palantir import PalantirChatGeneric


class TestGeneric(TestCase):
    model: MagicMock
    llm: PalantirChatGeneric
    using_live_model: bool

    def setUp(self) -> None:
        try:
            model = GenericChatCompletionLanguageModel.get("Llama_3_3_70b_Instruct")
            self.model = MagicMock(spec=GenericChatCompletionLanguageModel, wraps=model)
            self.using_live_model = True
            logging.info("Using live LLM")
        except Exception:
            logging.exception("Could not get live llm")
            self.model = MagicMock(spec=GenericChatCompletionLanguageModel)
            self.using_live_model = False

        self.llm = PalantirChatGeneric(model=self.model)

    def test_palantir_chat_generic(self) -> None:
        question = "Why is the sky blue?"
        if self.using_live_model is False:
            self.model.create_chat_completion.return_value = GenericChatCompletionResponse(
                completion="The sky is blue due to Rayleigh scattering of sunlight by the atmosphere.",
                token_usage=TokenUsage(
                    completion_tokens=10, max_tokens=15, prompt_tokens=5
                ),
            )

        answer = self.llm.invoke(question)

        self.model.create_chat_completion.assert_called_once()
        self.assertIn("rayleigh scattering", str(answer.content).lower())
