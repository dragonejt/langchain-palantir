"""Anthropic Claude chat model implementation for Palantir's Language Model Service.

This module provides a LangChain-compatible wrapper for Palantir's Anthropic Claude
chat models, enabling seamless integration with LangChain applications while leveraging
Palantir's hosted Anthropic Claude capabilities including multimodal input, tool calling,
and thinking capabilities.

Classes:
    PalantirChatAnthropic: LangChain chat model implementation using Palantir's
        Anthropic Claude models with full support for chat completions, multimodal input,
        tool binding, and advanced features like thinking mode.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    override,
)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from language_model_service_api.languagemodelservice_api import (
    ChatMessageRole,
)
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    ClaudeAnyToolChoice,
    ClaudeAutoToolChoice,
    ClaudeChatCompletionContent,
    ClaudeChatCompletionRequest,
    ClaudeChatCompletionResponse,
    ClaudeChatImageContent,
    ClaudeChatMessage,
    ClaudeChatMessageContent,
    ClaudeChatMessageRole,
    ClaudeChatTextContent,
    ClaudeChatToolResultContent,
    ClaudeChatToolUseContent,
    ClaudeCustomTool,
    ClaudeDisabledThinking,
    ClaudeEnabledThinking,
    ClaudeImageBase64Source,
    ClaudeImageSource,
    ClaudeImageUrlSource,
    ClaudeThinking,
    ClaudeTool,
    ClaudeToolChoice,
    ClaudeToolChoiceNoneToolChoice,
    ClaudeToolToolChoice,
)
from palantir_models.models import AnthropicClaudeLanguageModel
from pydantic import Field


class PalantirChatAnthropic(BaseChatModel):
    """LangChain ChatModel implementation using Palantir's Anthropic Claude models.

    This class provides a LangChain-compatible interface to Palantir's hosted Anthropic
    Claude models, supporting chat completions, multimodal input, tool usage, thinking
    mode, and all standard Claude parameters like temperature, max_tokens, etc.

    Attributes:
        model (AnthropicClaudeLanguageModel): AnthropicClaudeLanguageModel from
            palantir_models to use.
        temperature (Optional[float]): What sampling temperature to use.
        max_retries (int): Maximum number of retries to make when generating.
        top_k (Optional[int]): Only sample from the top K options for each subsequent token.
        top_p (Optional[float]): Total probability mass of tokens to consider at
            each step.
        thinking (Optional[dict[str, Any]]): Configuration for Claude's thinking mode.
            Should include type (enabled or disabled) and budget_tokens if enabled.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        stop (Optional[List[str]]): Default stop sequences.
    """

    model: AnthropicClaudeLanguageModel
    temperature: Optional[float] = None
    max_retries: int = 0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    thinking: Optional[dict[str, Any]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    stop: Optional[List[str]] = Field(default=None, alias="stop_sequences")

    @override
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completions for the given messages.

        This is the core method that handles the actual generation of responses
        by converting LangChain messages to Claude format, making the API call,
        and converting the response back to LangChain format.

        Args:
            messages (List[BaseMessage]): List of messages in the conversation.
                Can include HumanMessage, AIMessage, SystemMessage, ToolMessage,
                and FunctionMessage types.
            stop (Optional[List[str]]): Stop sequences to halt generation.
                Overrides instance-level stop sequences if provided.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager
                for handling generation events and logging.
            **kwargs (Any): Additional keyword arguments including tool_choice
                and tools for function calling.

        Returns:
            ChatResult: Result containing generated chat completions with usage
                metadata including token counts.

        Raises:
            ValueError: If an unsupported message type is provided.
            Exception: If the underlying model API call fails after retries.
        """
        request = ClaudeChatCompletionRequest(
            messages=list(map(self._convert_to_claude_chat_message, messages)),
            max_tokens=self.max_tokens,
            stop_sequences=self.stop,
            temperature=self.temperature,
            thinking=self._convert_to_claude_thinking(self.thinking),
            top_k=self.top_k,
            top_p=self.top_p,
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
        )
        response = self.model.create_chat_completion(
            completion_request=request, max_rate_limit_retries=self.max_retries
        )
        return ChatResult(
            generations=[self._convert_from_claude_chat_completion_response(response)]
        )

    @override
    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model for function calling capabilities.

        This method enables the model to call functions/tools during generation.
        Tools are converted to Claude's custom tool format and bound to the model
        instance, allowing for structured interactions and tool usage.

        Args:
            tools (Sequence[Union[dict[str, Any], type, Callable, BaseTool]]):
                Tools to bind to the model. Can be dictionaries with OpenAI
                function schemas, Python types, callable functions, or LangChain
                BaseTool instances.
            tool_choice (Optional[Union[dict, str, Literal["auto", "none", "any"], bool]]):
                Controls tool usage behavior. "auto" lets model decide, "none"
                disables tools, "any" forces any tool use, or specify a
                particular tool name.
            strict (Optional[bool]): Whether to use strict mode for function
                schema validation.
            parallel_tool_calls (Optional[bool]): Whether to allow multiple
                tool calls in a single response.
            **kwargs (Any): Additional keyword arguments passed to the parent
                bind method.

        Returns:
            Runnable[LanguageModelInput, BaseMessage]: A new runnable instance
                with tools bound, maintaining the same interface as the base
                chat model.
        """
        formatted_tools = []
        for tool in tools:
            openai_formatted = convert_to_openai_function(tool, strict=strict)
            formatted_tools.append(
                ClaudeTool(
                    custom=ClaudeCustomTool(
                        name=openai_formatted["name"],
                        description=openai_formatted["description"],
                        input_schema=openai_formatted["parameters"],
                    ),
                )
            )

        return super().bind(
            tools=formatted_tools,
            tool_choice=self._convert_to_claude_tool_choice(
                tool_choice, parallel_tool_calls
            ),
            strict=strict,
            **kwargs,
        )

    @property
    @override
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        Returns:
            Dict[str, Any]: Dictionary containing model identification parameters
                used for logging and monitoring purposes.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "PalantirChatAnthropic"
        }

    @property
    @override
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model.

        Returns:
            str: The language model type identifier used for logging purposes.
        """
        return "palantir-anthropic-chat"

    def _convert_to_claude_chat_message(
        self, message: BaseMessage
    ) -> ClaudeChatMessage:
        """Convert a LangChain BaseMessage to Claude ChatMessage format.

        Args:
            message (BaseMessage): LangChain message to convert. Supports
                HumanMessage, AIMessage, SystemMessage, ToolMessage, and
                FunctionMessage types.

        Returns:
            ClaudeChatMessage: Claude ChatMessage with appropriate role and content.
        """

        if isinstance(message, (HumanMessage, SystemMessage, FunctionMessage)):
            return ClaudeChatMessage(
                role=ClaudeChatMessageRole.USER,
                content=self._convert_to_claude_chat_message_content(message.content),
            )
        elif isinstance(message, AIMessage):
            if len(message.tool_calls) > 0:
                content = [
                    ClaudeChatMessageContent(
                        tool_use=ClaudeChatToolUseContent(
                            id=message.tool_calls[0]["id"],
                            name=message.tool_calls[0]["name"],
                            input=message.tool_calls[0]["args"],
                        )
                    )
                ]
            else:
                content = self._convert_to_claude_chat_message_content(message.content)
            return ClaudeChatMessage(
                role=ClaudeChatMessageRole.ASSISTANT,
                content=content,
            )
        elif isinstance(message, ToolMessage):
            return ClaudeChatMessage(
                role=ChatMessageRole.USER,
                content=[
                    ClaudeChatMessageContent(
                        tool_result=ClaudeChatToolResultContent(
                            content=message.text(),
                            tool_use_id=message.tool_call_id,
                            is_error=message.status == "error",
                        )
                    )
                ],
            )
        else:
            return ClaudeChatMessage(
                role=ClaudeChatMessageRole.UNKNOWN,
                content=self._convert_to_claude_chat_message_content(message.content),
            )

    def _convert_to_claude_chat_message_content(
        self, contents: Union[str, List[Union[str, dict]]]
    ) -> list[ClaudeChatMessageContent]:
        """Convert message content to Palantir ClaudeChatMessageContent format.

        Args:
            contents (Union[str, List[Union[str, dict]]]): Message content to
                convert. Can be a string, list of strings, or list of
                dictionaries with text or image content.

        Returns:
            list[ClaudeChatMessageContent]: List of formatted chat message content
                objects.

        Raises:
            ValueError: If an unsupported content type is encountered.
            TypeError: If content has an unsupported Python type.
        """

        if isinstance(contents, str):
            return [ClaudeChatMessageContent(text=ClaudeChatTextContent(text=contents))]
        formatted_content = []
        for content in contents:
            if isinstance(content, str):
                formatted_content.append(
                    ClaudeChatMessageContent(text=ClaudeChatTextContent(text=content))
                )
            elif isinstance(content, dict):
                if content["type"] == "text":
                    formatted_content.append(
                        ClaudeChatMessageContent(
                            text=ClaudeChatTextContent(text=content["text"])
                        )
                    )
                elif content["type"] == "image":
                    if content["source_type"] == "url":
                        source = ClaudeImageSource(
                            url=ClaudeImageUrlSource(url=content["image_url"])
                        )
                    elif content["source_type"] == "base64":
                        source = ClaudeImageSource(
                            base64=ClaudeImageBase64Source(
                                data=content["data"],
                                media_type=content["mime_type"]
                                .upper()
                                .replace("/", "_"),
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unsupported image source: {content['source_type']}"
                        )

                    formatted_content.append(
                        ClaudeChatMessageContent(
                            image=ClaudeChatImageContent(source=source)
                        )
                    )
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")
            else:
                raise TypeError(f"Unsupported content type: {type(content)}")
        return formatted_content

    def _convert_from_claude_chat_completion_response(
        self, response: ClaudeChatCompletionResponse
    ) -> ChatGeneration:
        """Convert Claude chat completion response to LangChain ChatGeneration format.

        Args:
            response (ClaudeChatCompletionResponse): The response from Palantir's
                Claude chat completion API.

        Returns:
            ChatGeneration: Chat generation with converted message and usage metadata.

        Raises:
            ValueError: If an unknown message role is encountered in the response.
        """
        text_content, tool_use_content = self._convert_to_claude_ai_message(
            response.content
        )

        if response.role == ClaudeChatMessageRole.USER:
            return ChatGeneration(message=HumanMessage(content=text_content))
        elif response.role == ClaudeChatMessageRole.ASSISTANT:
            return ChatGeneration(
                message=AIMessage(
                    content=text_content,
                    usage_metadata=UsageMetadata(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.max_tokens,
                    ),
                    tool_calls=tool_use_content,
                )
            )

        else:
            raise ValueError(f"Unknown message role: {response.role}")

    def _convert_to_claude_tool_choice(
        self,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none", "any"], bool]],
        parallel_tool_calls: Optional[bool],
    ) -> Optional[ClaudeToolChoice]:
        """Convert tool choice parameter to Claude ToolChoice format.

        Args:
            tool_choice (Optional[Union[dict, str, Literal["auto", "none", "any"], bool]]):
                Tool choice specification. Can be "auto", "none", "any",
                or a specific tool name.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
                When False, disables parallel tool use in Claude.

        Returns:
            Optional[ClaudeToolChoice]: Claude tool choice configuration,
                or None if no tool choice specified.
        """
        if tool_choice is None:
            return None
        elif tool_choice == "none":
            return ClaudeToolChoice(tool_choice_none=ClaudeToolChoiceNoneToolChoice())

        elif tool_choice == "auto":
            return ClaudeToolChoice(
                auto=ClaudeAutoToolChoice(
                    disable_parallel_tool_use=parallel_tool_calls is False
                )
            )

        elif tool_choice == "any":
            return ClaudeToolChoice(
                any=ClaudeAnyToolChoice(
                    disable_parallel_tool_use=parallel_tool_calls is False
                )
            )
        else:
            return ClaudeToolChoice(
                tool=ClaudeToolToolChoice(
                    name=tool_choice,
                    disable_parallel_tool_use=parallel_tool_calls is False,
                )
            )

    def _convert_to_claude_thinking(
        self, thinking: Optional[dict[str, Any]]
    ) -> Optional[ClaudeThinking]:
        """Convert thinking configuration to Claude Thinking format.

        Args:
            thinking (Optional[dict[str, Any]]): Thinking configuration dictionary.
                Should contain "type" key with "enabled" or "disabled" value.
                For enabled thinking, optionally include "budget_tokens".

        Returns:
            Optional[ClaudeThinking]: Claude thinking configuration, or None if
                no thinking configuration provided.
        """
        if thinking is not None and thinking.get("type") == "enabled":
            return ClaudeThinking(
                enabled=ClaudeEnabledThinking(budget_tokens=thinking["budget_tokens"])
            )
        elif thinking is not None and thinking.get("type") == "disabled":
            return ClaudeThinking(disabled=ClaudeDisabledThinking())
        else:
            return None

    def _convert_to_claude_ai_message(
        self, content_list: list[ClaudeChatCompletionContent]
    ) -> tuple[str | list[str], list[ToolCall]]:
        """Extract text content and tool calls from Claude completion content.

        Args:
            content_list (list[ClaudeChatCompletionContent]): List of content
                blocks from Claude's response, which can contain text and/or
                tool use information.

        Returns:
            tuple[str | list[str], list[ToolCall]]: A tuple containing:
                - Text content as a single string (if one item) or list of strings
                - List of ToolCall objects extracted from tool use content
        """
        text_content = []
        tool_use_content = []

        for content in content_list:
            if content.text is not None:
                text_content.append(content.text.text)
            elif content.tool_use is not None:
                tool_use_content.append(
                    ToolCall(
                        name=content.tool_use.name,
                        id=content.tool_use.id,
                        args=content.tool_use.input,
                    )
                )

        if len(text_content) == 1:
            text_content = text_content[0]
        return text_content, tool_use_content
