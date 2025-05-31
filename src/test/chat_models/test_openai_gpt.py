from typing import Optional
from unittest.mock import MagicMock
import logging
from datetime import datetime, timezone
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from language_model_service_api.languagemodelservice_api import ChatMessageRole
from palantir_models.models import OpenAiGptChatLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    GptChatCompletionChoice,
    GptChatCompletionChoiceMessage,
    GptChatCompletionResponse,
    GptChatUsage,
)
from langchain_core.tools import tool

from langchain_palantir import PalantirChatOpenAI


class TestOpenAiGpt:
    model: Optional[OpenAiGptChatLanguageModel]

    @classmethod
    def setup_class(cls) -> None:
        try:
            cls.model = OpenAiGptChatLanguageModel.get("GPT_4_1")
            logging.info("Using live LLM")
        except Exception as ex:
            logging.exception("Could not get LLM due to %s", ex)
            cls.model = None

    def test_palantir_chat_openai(self) -> None:
        if self.model == None:
            return
        question = "Why is the sky blue?"
        wrapped_llm = PalantirChatOpenAI(model=self.model)
        answer = wrapped_llm.invoke(question)

        assert len(answer.content) >= 0
        assert "rayleigh scattering" in answer.content.lower()

    def test_palantir_tool_calling(self) -> None:
        if self.model == None:
            return

        messages: list[BaseMessage] = [
            HumanMessage("Using the date_time tool, what is today's date?")
        ]

        @tool
        def date_time() -> datetime:
            """
            Returns the current datetime in python.
            Parameters: None
            """

            return datetime.now(timezone.utc)

        tools = {"date_time": date_time}

        llm = PalantirChatOpenAI(model=self.model)
        llm_with_tools = llm.bind_tools(tools.values())
        answer = llm_with_tools.invoke(messages)
        messages.append(answer)

        assert isinstance(answer, AIMessage)
        assert len(answer.tool_calls) == 1

        for tool_call in answer.tool_calls:
            messages.append(tools[tool_call["name"]].invoke(tool_call))

        final_answer = llm_with_tools.invoke(messages)

        date = datetime.now(timezone.utc)
        assert str(date.year) in final_answer.content
        assert str(date.day) in final_answer.content
