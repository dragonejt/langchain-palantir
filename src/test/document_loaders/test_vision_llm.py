import logging
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from palantir_models.models import (
    VisionLLMDocumentPageExtractor,
)

from langchain_palantir.document_loaders.vision_llm import PalantirVisionLLMLoader


class TestPalantirVisionLLMLoader(TestCase):
    model: VisionLLMDocumentPageExtractor
    using_live_model: bool

    def setUp(self) -> None:
        try:
            model = VisionLLMDocumentPageExtractor.get("GPT_4_1")
            self.model = MagicMock(spec=VisionLLMDocumentPageExtractor, wraps=model)
            self.using_live_model = True
        except Exception:
            logging.exception("Could not get live embeddings model")
            self.model = MagicMock(spec=VisionLLMDocumentPageExtractor)
            self.using_live_model = False

    def test_document_loader(self) -> None:
        loader = PalantirVisionLLMLoader(
            model=self.model,
            documents=[
                Path(__file__).parent.parent / "resources" / "a_tale_of_two_cities.jpg"
            ],
        )

        documents = loader.load()
        for document in documents:
            self.assertIn("best of times", document.page_content.lower())

    def test_document_lazy_loader(self) -> None:
        loader = PalantirVisionLLMLoader(
            model=self.model,
            documents=[
                Path(__file__).parent.parent / "resources" / "a_tale_of_two_cities.jpg"
            ],
        )

        documents = loader.lazy_load()
        for document in documents:
            self.assertIn("best of times", document.page_content.lower())
