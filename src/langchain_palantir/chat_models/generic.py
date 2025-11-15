"""Generic chat model implementation for Palantir's Language Model Service.

This module provides a LangChain-compatible wrapper for Palantir's generic chat
models, enabling seamless integration with LangChain applications while leveraging
Palantir's hosted generic language model capabilities for basic chat completions.

Classes:
    PalantirChatGeneric: LangChain chat model implementation using Palantir's
        generic chat models with support for basic chat completions and standard
        parameters like temperature and max_tokens.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith.schemas import UsageMetadata
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    DialogChatMessage,
    DialogRole,
    GenericChatCompletionRequest,
    GenericChatCompletionResponse,
)
from palantir_models.models import GenericChatCompletionLanguageModel
from pydantic import Field
from typing_extensions import override


class PalantirChatGeneric(BaseChatModel):
    """LangChain ChatModel implementation using Palantir's generic chat models.

    This class provides a LangChain-compatible interface to Palantir's hosted generic
    chat models, supporting basic chat completions and standard parameters like
    temperature, max_tokens, and stop sequences. This implementation is suitable
    for general-purpose conversational AI applications.

    Attributes:
        model (GenericChatCompletionLanguageModel): GenericChatCompletionLanguageModel from
            palantir_models to use.
        temperature (Optional[float]): What sampling temperature to use.
        max_retries (int): Maximum number of retries to make when generating.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        stop (Optional[List[str]]): Default stop sequences.
    """

    model: GenericChatCompletionLanguageModel
    temperature: Optional[float] = None
    max_retries: int = 0
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
        by converting LangChain messages to generic format, making the API call,
        and converting the response back to LangChain format.

        Args:
            messages (List[BaseMessage]): List of messages in the conversation.
                Can include HumanMessage, AIMessage, and SystemMessage types.
            stop (Optional[List[str]]): Stop sequences to halt generation.
                Overrides instance-level stop sequences if provided.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager
                for handling generation events and logging.
            **kwargs (Any): Additional keyword arguments for future extensibility.

        Returns:
            ChatResult: Result containing generated chat completions with usage
                metadata including token counts.

        Raises:
            Exception: If the underlying model API call fails after retries.
        """
        request = GenericChatCompletionRequest(
            messages=list(map(self._convert_to_generic_message, messages)),
            max_tokens=self.max_tokens,
            stop_sequences=self.stop,
            temperature=self.temperature,
        )

        response = self.model.create_chat_completion(
            completion_request=request, max_rate_limit_retries=self.max_retries
        )

        return ChatResult(
            generations=self._convert_from_generic_chat_completion_response(response)
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
            "model_name": "PalantirChatGeneric"
        }

    @property
    @override
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model.

        Returns:
            str: The language model type identifier used for logging purposes.
        """
        return "palantir-generic-chat"

    def _convert_to_generic_message(self, message: BaseMessage) -> DialogChatMessage:
        """Convert a LangChain BaseMessage to generic DialogChatMessage format.

        Args:
            message (BaseMessage): LangChain message to convert. Supports
                HumanMessage, AIMessage, and SystemMessage types. Other message
                types will be converted with UNKNOWN role.

        Returns:
            DialogChatMessage: Generic DialogChatMessage with appropriate role
                and content.
        """
        if isinstance(message, HumanMessage):
            return DialogChatMessage(
                role=DialogRole.USER,
                content=message.text,
            )
        elif isinstance(message, AIMessage):
            return DialogChatMessage(
                role=DialogRole.ASSISTANT,
                content=message.text,
            )
        elif isinstance(message, SystemMessage):
            return DialogChatMessage(
                role=DialogRole.SYSTEM,
                content=message.text,
            )
        else:
            return DialogChatMessage(
                role=DialogRole.UNKNOWN,
                content=message.text,
            )

    def _convert_from_generic_chat_completion_response(
        self, response: GenericChatCompletionResponse
    ) -> list[ChatGeneration]:
        """Convert generic chat completion response to LangChain ChatGeneration format.

        Args:
            response (GenericChatCompletionResponse): The response from Palantir's
                generic chat completion API containing the generated completion
                and token usage information.

        Returns:
            list[ChatGeneration]: List containing a single chat generation with
                the converted message and usage metadata including input, output,
                and total token counts.
        """
        return [
            ChatGeneration(
                message=AIMessage(
                    content=response.completion,
                    usage_metadata=UsageMetadata(
                        input_tokens=response.token_usage.prompt_tokens,
                        output_tokens=response.token_usage.completion_tokens,
                        total_tokens=response.token_usage.max_tokens,
                    ),
                )
            )
        ]
