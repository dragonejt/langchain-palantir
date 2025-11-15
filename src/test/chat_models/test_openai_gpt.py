import logging
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from language_model_service_api.languagemodelservice_api import ChatMessageRole
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    FunctionToolCallInfo,
    GptChatCompletionChoice,
    GptChatCompletionChoiceMessage,
    GptChatCompletionResponse,
    GptChatUsage,
    GptToolCall,
    GptToolCallInfo,
)
from palantir_models.models import (
    OpenAiGptChatLanguageModel,
    OpenAiGptChatWithVisionLanguageModel,
)

from langchain_palantir import PalantirChatOpenAI


class TestOpenAiGpt(TestCase):
    model: OpenAiGptChatLanguageModel
    llm: PalantirChatOpenAI
    using_live_model: bool

    def setUp(self) -> None:
        try:
            model = OpenAiGptChatWithVisionLanguageModel.get("GPT_4_1")
            self.model = MagicMock(
                spec=OpenAiGptChatWithVisionLanguageModel, wraps=model
            )
            self.using_live_model = True
            logging.info("Using live LLM")
        except Exception:
            logging.exception("Could not get live llm")
            self.model = MagicMock(spec=OpenAiGptChatWithVisionLanguageModel)
            self.using_live_model = False

        self.llm = PalantirChatOpenAI(model=self.model)

    def test_palantir_chat_openai(self) -> None:
        question = "Why is the sky blue?"
        if self.using_live_model is False:
            self.model.create_chat_completion.return_value = GptChatCompletionResponse(
                choices=[
                    GptChatCompletionChoice(
                        message=GptChatCompletionChoiceMessage(
                            role=ChatMessageRole.ASSISTANT,
                            content="The sky is blue due to Rayleigh scattering of sunlight by the atmosphere.",
                            tool_calls=[],
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ],
                created=int(datetime.now(timezone.utc).timestamp()),
                id="",
                model="GPT_4_1",
                object="chat.completion",
                usage=GptChatUsage(
                    completion_tokens=10, prompt_tokens=5, total_tokens=15
                ),
            )

        answer = self.llm.invoke(question)

        self.model.create_chat_completion.assert_called_once()
        self.assertIn("rayleigh scattering", answer.content.lower())

    def test_openai_tool_calling(self) -> None:
        messages = [HumanMessage("Using the date_time tool, what is today's date?")]

        @tool
        def date_time() -> str:
            """
            Returns the current datetime in ISO format.
            Parameters: None
            """

            return datetime.now(timezone.utc).isoformat()

        tools = {"date_time": date_time}
        llm_with_tools = self.llm.bind_tools(tools.values())

        if self.using_live_model is False:
            self.model.create_chat_completion.return_value = GptChatCompletionResponse(
                choices=[
                    GptChatCompletionChoice(
                        message=GptChatCompletionChoiceMessage(
                            role=ChatMessageRole.ASSISTANT,
                            tool_calls=[
                                GptToolCall(
                                    id="",
                                    tool_call=GptToolCallInfo(
                                        function=FunctionToolCallInfo(
                                            arguments="{}", name="date_time"
                                        )
                                    ),
                                )
                            ],
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ],
                created=int(datetime.now(timezone.utc).timestamp()),
                id="",
                model="GPT_4_1",
                object="chat.completion",
                usage=GptChatUsage(
                    completion_tokens=10, prompt_tokens=5, total_tokens=15
                ),
            )

        answer = llm_with_tools.invoke(messages)
        self.model.create_chat_completion.assert_called_once()
        self.assertIsInstance(answer, AIMessage)
        self.assertEqual(len(answer.tool_calls), 1)

        messages.append(answer)
        for tool_call in answer.tool_calls:
            messages.append(tools[tool_call["name"]].invoke(tool_call))

        final_answer = llm_with_tools.invoke(messages)
        self.assertEqual(self.model.create_chat_completion.call_count, 2)

        if self.using_live_model is True:
            date = datetime.now(timezone.utc)
            self.assertIn(str(date.year), final_answer.content)
            self.assertIn(str(date.day), final_answer.content)

    def test_openai_agent(self) -> None:
        if self.using_live_model is False:
            return

        messages = [HumanMessage("Using the date_time tool, what is today's date?")]

        @tool
        def date_time() -> str:
            """
            Returns the current datetime in ISO format.
            Parameters: None
            """

            return datetime.now(timezone.utc).isoformat()

        agent = create_agent(model=self.llm, tools=[date_time])

        answer = agent.invoke({"messages": messages})

        date = datetime.now(timezone.utc)
        self.assertIn(str(date.year), answer["messages"][-1].content)
        self.assertIn(str(date.day), answer["messages"][-1].content)

    def test_openai_vision(self) -> None:
        with open(
            Path(__file__).parent.parent / "resources" / "pizza.jpeg", "rb"
        ) as pizza_jpg:
            image_data = b64encode(pizza_jpg.read()).decode("utf-8")

            messages = [
                HumanMessage("What is in the following image?"),
                HumanMessage(
                    [
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image_data,
                            "mime_type": "image/jpeg",
                        }
                    ]
                ),
            ]

            if self.using_live_model is False:
                self.model.create_chat_completion.return_value = GptChatCompletionResponse(
                    choices=[
                        GptChatCompletionChoice(
                            message=GptChatCompletionChoiceMessage(
                                role=ChatMessageRole.ASSISTANT,
                                content="The image shows a freshly baked pizza topped with slices of pepperoni, black olives, melted cheese, and garnished with fresh basil leaves in the center.",
                                tool_calls=[],
                            ),
                            finish_reason="stop",
                            index=0,
                        )
                    ],
                    created=int(datetime.now(timezone.utc).timestamp()),
                    id="",
                    model="GPT_4_1",
                    object="chat.completion",
                    usage=GptChatUsage(
                        completion_tokens=10, prompt_tokens=5, total_tokens=15
                    ),
                )

            answer = self.llm.invoke(messages)

            self.model.create_chat_completion.assert_called_once()
            self.assertIn("pizza", answer.content.lower())
