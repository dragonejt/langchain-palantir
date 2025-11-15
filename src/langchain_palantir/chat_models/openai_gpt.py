"""OpenAI GPT chat model implementation for Palantir's Language Model Service.

This module provides a LangChain-compatible wrapper for Palantir's OpenAI GPT chat
models, enabling seamless integration with LangChain applications while leveraging
Palantir's hosted OpenAI GPT capabilities including multimodal inputs and tool calling.

Classes:
    PalantirChatOpenAI: LangChain chat model implementation using Palantir's
        OpenAI GPT models with full support for chat completions, multimodal inputs,
        and tool calling.
"""

from json import dumps, loads
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
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
    Base64ImageContent,
    ChatMessageContent,
    ChatMessageRole,
    MultiContentChatMessage,
)
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    FunctionToolCallInfo,
    GptAutoToolChoice,
    GptChatCompletionResponse,
    GptChatWithVisionCompletionRequest,
    GptFunctionTool,
    GptFunctionToolChoice,
    GptNoneToolChoice,
    GptRequiredToolChoice,
    GptSpecificToolChoice,
    GptTool,
    GptToolCall,
    GptToolCallInfo,
    GptToolChoice,
)
from palantir_models.models import OpenAiGptChatWithVisionLanguageModel
from pydantic import Field
from typing_extensions import override


class PalantirChatOpenAI(BaseChatModel):
    """LangChain ChatModel implementation using Palantir's OpenAI GPT models.

    This class provides a LangChain-compatible interface to Palantir's hosted OpenAI
    GPT models, supporting chat completions, multimodal inputs, tool calling, and all
    standard OpenAI parameters like temperature, max_tokens, etc.

    Attributes:
        model (OpenAiGptChatWithVisionLanguageModel): OpenAiGptChatWithVisionLanguageModel from
            palantir_models to use.
        temperature (Optional[float]): What sampling temperature to use.
        max_retries (int): Maximum number of retries to make when generating.
        presence_penalty (Optional[float]): Penalizes repeated tokens.
        frequency_penalty (Optional[float]): Penalizes repeated tokens according
            to frequency.
        seed (Optional[int]): Seed for generation.
        logit_bias (Optional[dict[int, float]]): Modify the likelihood of specified
            tokens appearing in the completion.
        n (Optional[int]): Number of chat completions to generate for each prompt.
        top_p (Optional[float]): Total probability mass of tokens to consider at
            each step.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        stop (Optional[List[str]]): Default stop sequences.
    """

    model: OpenAiGptChatWithVisionLanguageModel
    temperature: Optional[float] = None
    max_retries: int = 0
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    logit_bias: Optional[dict[int, float]] = None
    n: Optional[int] = None
    top_p: Optional[float] = None
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
        by converting LangChain messages to Palantir format, making the API call,
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
        request = GptChatWithVisionCompletionRequest(
            messages=list(map(self._convert_to_chat_message, messages)),
            frequency_penalty=self.frequency_penalty,
            logit_bias=self.logit_bias,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=stop or self.stop,
            temperature=self.temperature,
            tool_choice=kwargs.get("tool_choice"),
            tools=kwargs.get("tools"),
            top_p=self.top_p,
        )
        response = self.model.create_chat_completion(
            completion_request=request, max_rate_limit_retries=self.max_retries
        )
        return ChatResult(
            generations=self._convert_from_gpt_chat_completion_response(response)
        )

    @override
    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model for function calling capabilities.

        This method enables the model to call functions/tools during generation.
        Tools are converted to OpenAI function format and bound to the model
        instance, allowing for structured interactions and tool usage.

        Args:
            tools (Sequence[Union[dict[str, Any], type, Callable, BaseTool]]):
                Tools to bind to the model. Can be dictionaries with OpenAI
                function schemas, Python types, callable functions, or LangChain
                BaseTool instances.
            tool_choice (Optional[Union[dict, str, Literal["auto", "none", "required", "any"], bool]]):
                Controls tool usage behavior. "auto" lets model decide, "none"
                disables tools, "required" forces tool use, or specify a
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
        formatted_tools = [
            GptTool(
                function=GptFunctionTool(
                    **convert_to_openai_function(tool, strict=strict)
                )
            )
            for tool in tools
        ]
        return super().bind(
            tools=formatted_tools,
            tool_choice=self._convert_to_gpt_tool_choice(tool_choice),
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
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
            "model_name": "PalantirChatOpenAI"
        }

    @property
    @override
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model.

        Returns:
            str: The language model type identifier used for logging purposes.
        """
        return "palantir-openai-chat"

    def _convert_to_chat_message(self, message: BaseMessage) -> MultiContentChatMessage:
        """Convert a LangChain BaseMessage to Palantir ChatMessage format.

        Args:
            message (BaseMessage): LangChain message to convert. Supports
                HumanMessage, AIMessage, SystemMessage, ToolMessage, and
                FunctionMessage types.

        Returns:
            MultiContentChatMessage: Palantir MultiContentChatMessage with
                appropriate role and content.

        Raises:
            ValueError: If the message type is not supported.
        """
        if isinstance(message, HumanMessage):
            return MultiContentChatMessage(
                role=ChatMessageRole.USER,
                contents=self._convert_to_chat_message_content(message.content),
            )
        elif isinstance(message, AIMessage):
            return MultiContentChatMessage(
                role=ChatMessageRole.ASSISTANT,
                contents=self._convert_to_chat_message_content(message.content),
                tool_calls=list(
                    map(self._convert_to_gpt_tool_call, message.tool_calls)
                ),
            )
        elif isinstance(message, SystemMessage):
            return MultiContentChatMessage(
                role=ChatMessageRole.SYSTEM,
                contents=self._convert_to_chat_message_content(message.content),
            )
        elif isinstance(message, ToolMessage):
            return MultiContentChatMessage(
                role=ChatMessageRole.TOOL,
                contents=self._convert_to_chat_message_content(message.content),
                tool_call_id=message.tool_call_id,
            )
        elif isinstance(message, FunctionMessage):
            return MultiContentChatMessage(
                role=ChatMessageRole.FUNCTION,
                contents=self._convert_to_chat_message_content(message.content),
            )
        else:
            return MultiContentChatMessage(
                role=ChatMessageRole.UNKNOWN,
                contents=self._convert_to_chat_message_content(message.content),
            )

    def _convert_to_chat_message_content(
        self, contents: Union[str, List[Union[str, dict]]]
    ) -> list[ChatMessageContent]:
        """Convert message content to Palantir ChatMessageContent format.

        Args:
            contents (Union[str, List[Union[str, dict]]]): Message content to
                convert. Can be a string, list of strings, or list of
                dictionaries with text or image content.

        Returns:
            list[ChatMessageContent]: List of formatted chat message content
                objects.

        Raises:
            ValueError: If an unsupported content type is encountered.
            TypeError: If content has an unsupported Python type.
        """
        if isinstance(contents, str):
            return [ChatMessageContent(text=contents)]
        formatted_content = []
        for content in contents:
            if isinstance(content, str):
                formatted_content.append(ChatMessageContent(text=content))
            elif isinstance(content, dict):
                if content["type"] == "text":
                    formatted_content.append(ChatMessageContent(text=content["text"]))
                elif content["type"] == "image":
                    print(content.keys())
                    image_url = (
                        content.get("url")
                        or f"data:{content['mime_type']};base64,{content['base64']}"
                    )
                    formatted_content.append(
                        ChatMessageContent(
                            image=Base64ImageContent(image_url=image_url)
                        )
                    )
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")
            else:
                raise TypeError(f"Unsupported content type: {type(content)}")
        return formatted_content

    def _convert_from_gpt_chat_completion_response(
        self, response: GptChatCompletionResponse
    ) -> list[ChatGeneration]:
        """Convert Palantir GPT response to LangChain ChatGeneration format.

        Args:
            response (GptChatCompletionResponse): The response from Palantir's
                GPT chat completion API.

        Returns:
            list[ChatGeneration]: List of chat generations with converted
                messages and usage metadata.

        Raises:
            ValueError: If an unknown message role is encountered in the response.
        """
        generations = []
        for choice in response.choices:
            if choice.message.role == ChatMessageRole.USER:
                generations.append(
                    ChatGeneration(
                        message=HumanMessage(
                            content=choice.message.content,
                        )
                    )
                )
            elif choice.message.role == ChatMessageRole.ASSISTANT:
                generations.append(
                    ChatGeneration(
                        message=AIMessage(
                            content=choice.message.content or "",
                            usage_metadata=UsageMetadata(
                                input_tokens=response.usage.prompt_tokens,
                                output_tokens=response.usage.completion_tokens,
                                total_tokens=response.usage.total_tokens,
                            ),
                            tool_calls=list(
                                map(
                                    self._convert_from_gpt_tool_call,
                                    choice.message.tool_calls,
                                )
                            ),
                        )
                    )
                )
            elif choice.message.role == ChatMessageRole.SYSTEM:
                generations.append(
                    ChatGeneration(
                        message=SystemMessage(
                            content=choice.message.content or "",
                        )
                    )
                )
            elif choice.message.role == ChatMessageRole.TOOL:
                generations.append(
                    ChatGeneration(
                        message=ToolMessage(
                            content=choice.message.content or "",
                        )
                    )
                )
            elif choice.message.role == ChatMessageRole.FUNCTION:
                generations.append(
                    ChatGeneration(
                        message=FunctionMessage(content=choice.message.content or "")
                    )
                )
            else:
                raise ValueError(f"Unknown message role: {choice.message.role}")

        return generations

    def _convert_to_gpt_tool_call(self, tool_call: ToolCall) -> GptToolCall:
        """Convert LangChain ToolCall to Palantir GptToolCall format.

        Args:
            tool_call (ToolCall): LangChain tool call to convert.

        Returns:
            GptToolCall: Palantir GPT tool call with serialized arguments.
        """
        return GptToolCall(
            id=tool_call["id"],
            tool_call=GptToolCallInfo(
                function=FunctionToolCallInfo(
                    name=tool_call["name"], arguments=dumps(tool_call["args"])
                )
            ),
        )

    def _convert_from_gpt_tool_call(self, gpt_tool_call: GptToolCall) -> ToolCall:
        """Convert Palantir GptToolCall to LangChain ToolCall format.

        Args:
            gpt_tool_call (GptToolCall): Palantir GPT tool call to convert.

        Returns:
            ToolCall: LangChain tool call with deserialized arguments.
        """
        return ToolCall(
            name=gpt_tool_call.tool_call.function.name,
            args=loads(gpt_tool_call.tool_call.function.arguments),
            id=gpt_tool_call.id,
            type=gpt_tool_call.tool_call.type,
        )

    def _convert_to_gpt_tool_choice(
        self,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ],
    ) -> Optional[GptToolChoice]:
        """Convert tool choice parameter to Palantir GptToolChoice format.

        Args:
            tool_choice (Optional[Union[dict, str, Literal["auto", "none", "required", "any"], bool]]):
                Tool choice specification. Can be "auto", "none", "required",
                or a specific tool name.

        Returns:
            Optional[GptToolChoice]: Palantir GPT tool choice configuration,
                or None if no tool choice specified.
        """
        if tool_choice is None:
            return None
        elif tool_choice == "none":
            return GptToolChoice(none=GptNoneToolChoice())
        elif tool_choice == "auto":
            return GptToolChoice(auto=GptAutoToolChoice())
        elif tool_choice == "required":
            return GptToolChoice(required=GptRequiredToolChoice())
        else:
            return GptToolChoice(
                specific=GptSpecificToolChoice(
                    function=GptFunctionToolChoice(name=tool_choice)
                )
            )
