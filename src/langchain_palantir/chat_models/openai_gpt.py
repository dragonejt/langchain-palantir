from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from typing_extensions import override
from json import loads, dumps
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_openai import BaseChatOpenAI

from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    GenerationChunk,
)
import logging
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from palantir_models.transforms import OpenAiGptChatLanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
    convert_to_openai_function,
)
from palantir_models.models import OpenAiGptChatLanguageModel
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    FunctionToolCallInfo,
    GptAutoToolChoice,
    GptChatCompletionRequest,
    GptChatCompletionResponse,
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
from language_model_service_api.languagemodelservice_api import (
    ChatMessage,
    ChatMessageRole,
    FunctionCall,
)


class PalantirChatOpenAI(BaseChatOpenAI):
    model: OpenAiGptChatLanguageModel
    temperature: float = 0

    @override
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = GptChatCompletionRequest(
            messages=list(map(self._convert_to_gpt_chat_completion_request, messages)),
            temperature=self.temperature,
            stop=stop,
            tools=kwargs.get("tools"),
            # tool_choice=kwargs.get("tool_choice"),
        )
        response = self.model.create_chat_completion(request)
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
        formatted_tools = list(
            map(
                lambda tool: GptTool(
                    function=GptFunctionTool(
                        **convert_to_openai_function(tool, strict=strict)
                    )
                ),
                tools,
            )
        )

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
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "PalantirChatOpenAI",
        }

    @property
    @override
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"

    def _convert_to_gpt_chat_completion_request(
        self, message: BaseMessage
    ) -> ChatMessage:
        if isinstance(message, HumanMessage):
            return ChatMessage(
                name=message.name,
                role=ChatMessageRole.USER,
                content=str(message.content),
            )
        elif isinstance(message, AIMessage):
            return ChatMessage(
                name=message.name,
                role=ChatMessageRole.ASSISTANT,
                content=str(message.content),
                tool_calls=list(
                    map(self._convert_to_gpt_tool_call, message.tool_calls)
                ),
            )
        elif isinstance(message, SystemMessage):
            return ChatMessage(
                name=message.name,
                role=ChatMessageRole.SYSTEM,
                content=str(message.content),
            )
        elif isinstance(message, ToolMessage):
            return ChatMessage(
                name=message.name,
                role=ChatMessageRole.TOOL,
                content=str(message.content),
                tool_call_id=message.tool_call_id,
            )
        elif isinstance(message, FunctionMessage):
            return ChatMessage(ChatMessageRole.FUNCTION, str(message.content))
        else:
            raise ValueError(f"Unknown message type: {message.type}")

    def _convert_from_gpt_chat_completion_response(
        self, response: GptChatCompletionResponse
    ) -> list[ChatGeneration]:
        usage_metadata = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        generations = []
        for choice in response.choices:
            if choice.message.role == ChatMessageRole.USER:
                generations.append(
                    ChatGeneration(
                        message=HumanMessage(
                            content=choice.message.content or "",
                            usage_metadata=usage_metadata,
                        )
                    )
                )
            elif choice.message.role == ChatMessageRole.ASSISTANT:
                generations.append(
                    ChatGeneration(
                        message=AIMessage(
                            content=choice.message.content or "",
                            usage_metadata=usage_metadata,
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
                            usage_metadata=usage_metadata,
                        )
                    )
                )
            elif choice.message.role == ChatMessageRole.TOOL:
                generations.append(
                    ChatGeneration(
                        message=ToolMessage(
                            content=choice.message.content or "",
                            usage_metadata=usage_metadata,
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
        return GptToolCall(
            id=tool_call["id"],
            tool_call=GptToolCallInfo(
                function=FunctionToolCallInfo(
                    name=tool_call["name"], arguments=dumps(tool_call["args"])
                )
            ),
        )

    def _convert_from_gpt_tool_call(self, gpt_tool_call: GptToolCall) -> ToolCall:
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
    ) -> GptToolChoice:
        if tool_choice == "none":
            return GptToolChoice(none=GptNoneToolChoice())
        elif tool_choice == "auto":
            return GptToolChoice(auto=GptAutoToolChoice())
        elif tool_choice == "require":
            return GptToolChoice(required=GptRequiredToolChoice())
        else:
            return GptToolChoice(
                specific=GptSpecificToolChoice(
                    function=GptFunctionToolChoice(name=tool_choice)
                )
            )
