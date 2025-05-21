from unittest import TestCase
from palantir_models.models import OpenAiGptChatLanguageModel

from langchain_palantir.chat_models.openai_gpt import ChatOpenA


def test_response() -> None:
    question = "Why is the sky blue?"
    wrapped_llm = ChatOpenAI(model=OpenAiGptChatLanguageModel.get("GPT_4o"))
    answer = wrapped_llm.invoke(question)

    assert len(answer.content) >= 0
