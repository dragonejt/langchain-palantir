from unittest.mock import MagicMock
import logging
from language_model_service_api.languagemodelservice_api import ChatMessageRole
from palantir_models.models import OpenAiGptChatLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    GptChatCompletionChoice,
    GptChatCompletionChoiceMessage,
    GptChatCompletionResponse,
    GptChatUsage,
)

from langchain_palantir.chat_models.openai_gpt import PalantirChatOpenAI


class TestOpenAiGpt:
    model: OpenAiGptChatLanguageModel

    @classmethod
    def setup_class(cls) -> None:
        try:
            cls.model = OpenAiGptChatLanguageModel.get("GPT_4o")
            logging.info("Using live LLM")
        except Exception as ex:
            logging.warning(f"Using mock LLM due to {ex}")
            mock_choice = GptChatCompletionChoice(
                finish_reason="",
                index=0,
                message=GptChatCompletionChoiceMessage(
                    role=ChatMessageRole.ASSISTANT,
                    content="The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it is made up of many different colors, each with different wavelengths. Blue light has a shorter wavelength compared to other colors like red or yellow.\n\nAs sunlight passes through the atmosphere, it interacts with air molecules and small particles. Shorter wavelengths of light, such as blue, are scattered in all directions by these molecules and particles. This scattering causes the blue light to be more prominent and visible from all directions, making the sky appear blue to our eyes.\n\nDuring sunrise and sunset, the sky can appear red or orange because the sunlight has to pass through a greater thickness of the atmosphere. This causes more scattering of the shorter wavelengths and allows the longer wavelengths, like red and orange, to dominate the sky's appearance.",
                ),
            )

            mock_response = GptChatCompletionResponse(
                choices=[mock_choice],
                created=0,
                id="",
                model="GPT_4o",
                object="",
                usage=GptChatUsage(10, 15, 25),
            )

            mock_model = MagicMock()
            mock_model.create_chat_completion.return_value = mock_response

            cls.model = mock_model

    def test_response(self) -> None:
        question = "Why is the sky blue?"
        wrapped_llm = PalantirChatOpenAI(model=self.model)
        answer = wrapped_llm.invoke(question)

        assert len(answer.content) >= 0
        assert "rayleigh scattering" in answer.content.lower()
