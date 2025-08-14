import logging
from time import time
from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock
import mlflow
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
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
from palantir_models.models import OpenAiGptChatLanguageModel
from palantir_models.code_workspaces import ModelOutput
from langchain_palantir import PalantirChatOpenAI


class TestOpenAiGpt(TestCase):
    model: OpenAiGptChatLanguageModel
    llm: PalantirChatOpenAI
    using_live_model: bool

    def setUp(self) -> None:
        try:
            model = OpenAiGptChatLanguageModel.get("GPT_4_1")
            self.model = MagicMock(spec=OpenAiGptChatLanguageModel, wraps=model)
            self.using_live_model = True
            logging.info("Using live LLM")
        except Exception:
            logging.exception("Could not get live llm")
            self.model = MagicMock(spec=OpenAiGptChatLanguageModel)
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

    def test_palantir_tool_calling(self) -> None:
        messages: list[BaseMessage] = [
            HumanMessage("Using the date_time tool, what is today's date?")
        ]

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

    def test_palantir_mlflow(self) -> None:
        if self.using_live_model is False:
            return

        model_output = ModelOutput("langchain-palantir")

        experiment = model_output.create_experiment(name=f"langchain-palantir-{time()}")
        with experiment.as_mlflow_run():
            mlflow.langchain.autolog()
            messages: list[BaseMessage] = [
                HumanMessage("Using the date_time tool, what is today's date?")
            ]

            @tool
            def date_time() -> str:
                """
                Returns the current datetime in ISO format.
                Parameters: None
                """

                return datetime.now(timezone.utc).isoformat()

            tools = {"date_time": date_time}
            llm_with_tools = self.llm.bind_tools(tools.values())

            answer = llm_with_tools.invoke(messages)
            self.model.create_chat_completion.assert_called_once()
            self.assertIsInstance(answer, AIMessage)
            self.assertEqual(len(answer.tool_calls), 1)

            messages.append(answer)
            for tool_call in answer.tool_calls:
                messages.append(tools[tool_call["name"]].invoke(tool_call))

            final_answer = llm_with_tools.invoke(messages)
            self.assertEqual(self.model.create_chat_completion.call_count, 2)

        model_output.publish(experiment)
