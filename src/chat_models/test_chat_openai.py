from unittest import TestCase

from chat_models.chat_openai import ChatOpenAI

def test_response() -> None:
        question = "Why is the sky blue?"
        wrapped_llm = ChatOpenAI(model="GPT_4o")
        answer = wrapped_llm.invoke(question)

        print(answer)