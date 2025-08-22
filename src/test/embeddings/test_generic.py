import logging
from unittest import TestCase
from unittest.mock import MagicMock

from language_model_service_api.languagemodelservice_api_embeddings_v3 import (
    EmbeddingUsage,
    GenericEmbeddingsResponse,
)
from palantir_models.models import GenericEmbeddingModel

from langchain_palantir import PalantirGenericEmbeddings


class TestPalantirGenericEmbeddings(TestCase):
    model: GenericEmbeddingModel
    embedding: PalantirGenericEmbeddings
    using_live_model: bool

    def setUp(self) -> None:
        try:
            model = GenericEmbeddingModel.get("Text_Embedding_3_Small")
            self.model = MagicMock(spec=GenericEmbeddingModel, wraps=model)
            self.using_live_model = True
        except Exception:
            logging.exception("Could not get live embeddings model")
            self.model = MagicMock(spec=GenericEmbeddingModel)
            self.using_live_model = False

        self.embedding = PalantirGenericEmbeddings(model=self.model)

    def test_embed_query(self) -> None:
        if self.using_live_model is False:
            self.model.create_embeddings.return_value = GenericEmbeddingsResponse(
                embeddings=[[0, 0, 0]], usage=EmbeddingUsage(input_tokens=3)
            )

        embeddings = self.embedding.embed_query("Hello World")

        self.model.create_embeddings.assert_called_once()
        self.assertGreater(len(embeddings), 0)

    def test_embed_documents(self) -> None:
        texts = ["Hello World", "Hello AI"]
        if self.using_live_model is False:
            self.model.create_embeddings.return_value = GenericEmbeddingsResponse(
                embeddings=[[0 for token in text] for text in texts],
                usage=EmbeddingUsage(input_tokens=4),
            )

        embeddings = self.embedding.embed_documents(texts)
        self.model.create_embeddings.assert_called_once()
        for embedding in embeddings:
            self.assertGreater(len(embedding), 0)
