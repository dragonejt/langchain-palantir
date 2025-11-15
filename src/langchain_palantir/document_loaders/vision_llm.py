"""LangChain document loader for Palantir Vision LLM models.

This module provides a LangChain-compatible document loader that uses Palantir's
Vision LLM models to extract text content from images and documents. It enables
multimodal document processing by converting visual content (images, PDFs, scanned
documents) into text format suitable for RAG pipelines and other LangChain workflows.

The loader leverages Palantir's VisionLLMDocumentPageExtractor models to perform
optical character recognition (OCR) and document understanding on image files or
raw bytes. It supports both eager and lazy loading patterns, making it efficient
for processing large batches of documents.

Key Features:
    - Extract text from images using Palantir Vision LLM models
    - Support for multiple image formats (PNG, JPEG, etc.)
    - Process documents from file paths or raw bytes
    - Integration with LangChain's document processing pipeline

Classes:
    PalantirVisionLLMLoader: Document loader using Palantir Vision LLM models.
"""

import mimetypes
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from palantir_models.models import VisionLLMDocumentPageExtractor
from pydantic import Field
from typing_extensions import Iterator, Optional, Union, override


class PalantirVisionLLMLoader(BaseLoader):
    """LangChain document loader that extracts text from images using Vision LLM.

    This loader uses Palantir's Vision LLM models to perform document understanding
    and text extraction from image files. It processes images through a Vision LLM
    model that can recognize text, understand document structure, and extract
    meaningful content from visual inputs like scanned documents, photos, or PDFs.

    The loader supports both file path inputs and raw byte inputs, automatically
    detecting MIME types for files and preserving metadata. It implements both
    eager loading (load all at once) and lazy loading (stream one at a time)
    patterns for flexibility with different workload sizes.

    Attributes:
        model: The Palantir Vision LLM model used for text extraction.
        documents: List of documents to process, either as file paths or raw bytes.
        max_retries: Maximum number of retry attempts for model inference. Defaults to 0.
        max_tokens: Optional maximum number of tokens to generate in the extraction.
            If None, uses the model's default limit.
    """

    model: VisionLLMDocumentPageExtractor
    documents: list[Union[Path, bytes]]
    max_retries: int = 0
    max_tokens: Optional[int] = Field(default=None)

    def __init__(
        self,
        model: VisionLLMDocumentPageExtractor,
        documents: list[Union[Path, bytes]],
        *,
        max_retries: int = 0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initializes the Palantir Vision LLM document loader.

        Args:
            model: The VisionLLMDocumentPageExtractor model instance to use for
                text extraction from images.
            documents: List of documents to process. Each document can be either
                a Path object pointing to an image file, or raw bytes representing
                image data.
            max_retries: Maximum number of retry attempts if model inference fails.
                Defaults to 0 (no retries).
            max_tokens: Optional maximum number of tokens to generate during
                extraction. If None, uses the model's default token limit.
        """
        self.model = model
        self.documents = documents
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    @override
    def load(self) -> list[Document]:
        """Loads all documents and extracts text from them.

        Processes all documents in the documents list by extracting text content
        from each image using the Vision LLM model. This method loads all documents
        eagerly into memory.

        Returns:
            List of Document objects containing extracted text in page_content and
            metadata including source file name and MIME type (for file-based inputs).
        """
        return list(map(self._load_one, self.documents))

    @override
    def lazy_load(self) -> Iterator[Document]:
        """Lazily loads and processes documents one at a time.

        Returns an iterator that processes documents on-demand rather than loading
        all at once. This is more memory-efficient for large document collections
        as it only loads and processes one document at a time.

        Yields:
            Document objects containing extracted text in page_content and metadata.
        """
        return map(self._load_one, self.documents)

    def _load_one(self, document: Union[Path, bytes]) -> Document:
        """Loads and processes a single document through the Vision LLM model.

        Handles both file path and raw bytes inputs. For file paths, automatically
        detects the MIME type and reads the file contents. Converts the image to
        the format expected by the Vision LLM model and extracts text content.

        Args:
            document: Either a Path object pointing to an image file, or raw bytes
                representing image data.

        Returns:
            Document object with:
                - page_content: The extracted text from the image
                - metadata: Dictionary containing source filename and MIME type
                  (for file-based inputs) or empty (for bytes inputs)
        """
        image_bytes: bytes = document
        mime_type = "image/png"
        metadata = dict()
        if isinstance(document, Path):
            mime_type = mimetypes.guess_type(document)[0]
            metadata = {"source": document.name, "mime_type": mime_type}
            with open(document, "rb") as image:
                image_bytes = image.read()

        response = self.model.create_extraction(
            image_bytes,
            mime_type.upper().replace("/", "_"),
            max_rate_limit_retries=self.max_retries,
            max_tokens=self.max_tokens,
        )

        return Document(page_content=response, metadata=metadata)
