from typing import override

from langchain_core.embeddings import Embeddings
from language_model_service_api.languagemodelservice_api_embeddings_v3 import (
    EmbeddingInputType,
    GenericEmbeddingsRequest,
    QueryEmbeddingContent,
)
from palantir_models.models import GenericEmbeddingModel


class PalantirGenericEmbeddings(Embeddings):
    model: GenericEmbeddingModel

    def __init__(self, model: GenericEmbeddingModel):
        self.model = model

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Call out to Palantir's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        request = GenericEmbeddingsRequest(inputs=texts)
        response = self.model.create_embeddings(request)

        return response.embeddings

    @override
    def embed_query(self, text: str) -> list[float]:
        """
        Call out to Palantir's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        request = GenericEmbeddingsRequest(
            inputs=[text], input_type=EmbeddingInputType(query=QueryEmbeddingContent())
        )
        response = self.model.create_embeddings(request)

        return response.embeddings[0]
