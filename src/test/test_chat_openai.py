from unittest import TestCase
from palantir_models.models import OpenAiGptChatLanguageModel

from langchain_palantir.chat_models.openai_gpt import PalantirChatOpenAI


def test_response() -> str:
    question = "Why is the sky blue?"
    model = OpenAiGptChatLanguageModel.get("GPT_4o")
    wrapped_llm = PalantirChatOpenAI(model=model)
    answer = wrapped_llm.invoke(question)

    assert len(answer.content) >= 0

    return answer.content
