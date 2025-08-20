"""Generic embeddings implementation for Palantir's Language Model Service.

This module provides a LangChain-compatible wrapper for Palantir's generic embedding
models, allowing seamless integration with LangChain applications while leveraging
Palantir's embedding capabilities.

Classes:
    PalantirGenericEmbeddings: LangChain embeddings implementation using Palantir's
        generic embedding models.
"""

from typing import override

from langchain_core.embeddings import Embeddings
from language_model_service_api.languagemodelservice_api_embeddings_v3 import (
    EmbeddingInputType,
    GenericEmbeddingsRequest,
    QueryEmbeddingContent,
)
from palantir_models.models import GenericEmbeddingModel


class PalantirGenericEmbeddings(Embeddings):
    """LangChain embeddings implementation using Palantir's generic embedding models.

    This class provides a LangChain-compatible interface to Palantir's generic
    embedding models, supporting both document and query embeddings with appropriate
    input type handling.

    Attributes:
        model (GenericEmbeddingModel): The Palantir generic embedding model instance
            used for generating embeddings.
    """

    model: GenericEmbeddingModel

    def __init__(self, model: GenericEmbeddingModel):
        """Initialize the embeddings wrapper with a Palantir generic embedding model.

        Args:
            model (GenericEmbeddingModel): A configured Palantir generic embedding
                model instance.
        """
        self.model = model

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents using Palantir's embedding endpoint.

        This method is optimized for embedding documents that will be stored in a
        vector database or used for similarity search. The embeddings are generated
        using the default document input type.

        Args:
            texts (list[str]): A list of document strings to embed. Each string
                represents a document or text chunk.

        Returns:
            list[list[float]]: A list of embedding vectors, where each vector is a
                list of floats representing the embedding for the corresponding input
                text. The order of embeddings matches the order of input texts.

        Raises:
            Exception: If the embedding service returns an error or if the model
                is not properly configured.
        """
        request = GenericEmbeddingsRequest(inputs=texts)
        response = self.model.create_embeddings(request)

        return response.embeddings

    @override
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query using Palantir's embedding endpoint.

        This method is specifically optimized for embedding search queries that will
        be used to find similar documents. It uses the query input type to ensure
        optimal embedding generation for search scenarios.

        Args:
            text (str): The search query string to embed.

        Returns:
            list[float]: An embedding vector as a list of floats representing the
                query embedding.

        Raises:
            Exception: If the embedding service returns an error or if the model
                is not properly configured.
        """
        request = GenericEmbeddingsRequest(
            inputs=[text], input_type=EmbeddingInputType(query=QueryEmbeddingContent())
        )
        response = self.model.create_embeddings(request)

        return response.embeddings[0]
