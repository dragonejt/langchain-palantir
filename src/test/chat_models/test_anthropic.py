import logging
from base64 import b64encode
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    ClaudeChatCompletionContent,
    ClaudeChatCompletionResponse,
    ClaudeChatCompletionTextContent,
    ClaudeChatCompletionToolUseContent,
    ClaudeChatMessageRole,
    ClaudeTokenUsage,
)
from palantir_models.models import (
    AnthropicClaudeLanguageModel,
)

from langchain_palantir import PalantirChatAnthropic


class TestAnthropic(TestCase):
    model: MagicMock
    llm: PalantirChatAnthropic
    using_live_model: bool

    def setUp(self) -> None:
        try:
            model = AnthropicClaudeLanguageModel.get("AnthropicClaude_4_Sonnet")
            print(model._model_api_name)
            self.model = MagicMock(spec=AnthropicClaudeLanguageModel, wraps=model)
            self.using_live_model = True
            logging.info("Using live LLM")
        except Exception:
            logging.exception("Could not get live llm")
            self.model = MagicMock(spec=AnthropicClaudeLanguageModel)
            self.using_live_model = False

        self.llm = PalantirChatAnthropic(model=self.model)

    def test_palantir_chat_anthropic(self) -> None:
        question = "Why is the sky blue?"
        if self.using_live_model is False:
            self.model.create_chat_completion.return_value = ClaudeChatCompletionResponse(
                content=[
                    ClaudeChatCompletionContent(
                        text=ClaudeChatCompletionTextContent(
                            text="The sky is blue due to Rayleigh scattering of sunlight by the atmosphere."
                        )
                    )
                ],
                id="",
                model="AnthropicClaude_4_Sonnet",
                role=ClaudeChatMessageRole.ASSISTANT,
                usage=ClaudeTokenUsage(input_tokens=5, output_tokens=10, max_tokens=15),
            )

        answer = self.llm.invoke(question)

        self.model.create_chat_completion.assert_called_once()
        self.assertIn("rayleigh scattering", str(answer.content).lower())

    def test_anthropic_tool_calling(self) -> None:
        messages = [HumanMessage("Using the date_time tool, what is today's date?")]

        @tool
        def date_time() -> str:
            """
            Returns the current datetime in ISO format.
            Parameters: None
            """

            return datetime.now(timezone.utc).isoformat()

        tools = {"date_time": date_time}
        llm_with_tools = self.llm.bind_tools(list(tools.values()))

        if self.using_live_model is False:
            self.model.create_chat_completion.return_value = (
                ClaudeChatCompletionResponse(
                    content=[
                        ClaudeChatCompletionContent(
                            tool_use=ClaudeChatCompletionToolUseContent(
                                id="", name="date_time", input={}
                            )
                        )
                    ],
                    id="",
                    model="AnthropicClaude_4_Sonnet",
                    role=ClaudeChatMessageRole.ASSISTANT,
                    usage=ClaudeTokenUsage(
                        input_tokens=5, output_tokens=10, max_tokens=15
                    ),
                )
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
            self.assertIn(str(date.year), str(final_answer.content))
            self.assertIn(str(date.day), str(final_answer.content))

    def test_anthropic_agent(self) -> None:
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

    def test_anthropic_vision(self) -> None:
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
                self.model.create_chat_completion.return_value = ClaudeChatCompletionResponse(
                    content=[
                        ClaudeChatCompletionContent(
                            text=ClaudeChatCompletionTextContent(
                                text="The image shows a freshly baked pizza topped with slices of pepperoni, black olives, melted cheese, and garnished with fresh basil leaves in the center."
                            )
                        )
                    ],
                    id="",
                    model="AnthropicClaude_4_Sonnet",
                    role=ClaudeChatMessageRole.ASSISTANT,
                    usage=ClaudeTokenUsage(
                        input_tokens=5, output_tokens=10, max_tokens=15
                    ),
                )

            answer = self.llm.invoke(messages)

            self.model.create_chat_completion.assert_called_once()
            self.assertIn("pizza", str(answer.content).lower())
