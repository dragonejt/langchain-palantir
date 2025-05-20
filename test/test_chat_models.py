from unittest import TestCase

class TestChatOpenAI(TestCase):

    def test_response(self) -> None:
        question = "Why is the sky blue?"
        wrapped_llm = CustomLLM(model=model)
        answer = wrapped_llm.invoke(question)