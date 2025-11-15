"""LangChain vector store integration for Palantir Foundry Ontology.

This module provides a LangChain-compatible vector store implementation that uses
Palantir Foundry's Ontology as the backend storage system. It enables semantic
search and retrieval-augmented generation (RAG) workflows by storing document
embeddings as Ontology objects with vector properties.

The implementation leverages Foundry's native vector similarity search capabilities
through the nearest_neighbors API, allowing efficient retrieval of semantically
similar documents. Documents are stored as Ontology objects with their text content,
metadata (as JSON), and vector embeddings managed through Ontology actions.

Key Features:
    - Store and retrieve document embeddings in Foundry Ontology
    - Perform semantic similarity search using vector properties
    - Manage documents via Ontology actions (create, delete)

Classes:
    OntologyActions: Container for create and delete Ontology action types.
    PalantirOntology: LangChain vector store backed by Foundry Ontology.
"""

import json

from foundry_sdk_runtime.ontology_object_set import OntologyObjectSet
from foundry_sdk_runtime.properties.property_types import VectorObjectTypeProperty
from foundry_sdk_runtime.types import (
    ActionConfig,
    ActionMode,
    ReturnEditsMode,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStore
from typing_extensions import Any, Sequence, override


class OntologyActions:
    """Container for Ontology action types used for vector store operations.

    This class holds references to Ontology action types that enable
    creating and deleting vector objects in a Palantir Foundry Ontology.

    Attributes:
        create: Action type for creating new vector objects in the Ontology.
        delete: Action type for deleting existing vector objects from the Ontology.
    """

    def __init__(self, create, delete) -> None:
        """Initializes the OntologyActions container.

        Args:
            create: Ontology action type for creating vector objects.
            delete: Ontology action type for deleting vector objects.
        """
        self.create = create
        self.delete = delete


class PalantirOntology(VectorStore):
    """LangChain vector store implementation backed by Palantir Foundry Ontology.

    This class provides a vector store interface for storing and retrieving
    document embeddings using Palantir Foundry's Ontology as the backend storage.
    It supports similarity search, document addition, deletion, and retrieval
    operations through Ontology actions and object sets.

    The vector store uses Ontology's native vector property support and nearest
    neighbor search to enable efficient semantic similarity queries. Documents are
    persisted as Ontology objects, with embeddings stored in vector properties and
    metadata serialized as JSON strings.

    Attributes:
        vector_store: The Ontology object set used to store vector objects.
        vector_property: The vector property type used for similarity search.
        actions: Container holding create and delete action types.
        embedding: The embeddings model used to generate vector representations.
    """

    def __init__(
        self,
        vector_store: OntologyObjectSet,
        vector_property: VectorObjectTypeProperty,
        actions: OntologyActions,
        embedding: Embeddings,
    ) -> None:
        """Initializes the Palantir Ontology vector store.
        The dimensions of the vector property must match the output dimension
        of the Embeddings model!

        Args:
            vector_store: Ontology object set for storing and querying vector objects.
            vector_property: Vector property type used for nearest neighbor search.
            actions: Container with create and delete action types.
            embedding: Embeddings model for generating vector representations.
        """
        self.vector_store = vector_store
        self.vector_property = vector_property
        self.actions = actions
        self.embedding = embedding

    @property
    @override
    def embeddings(self) -> Embeddings:
        """Returns the embeddings model used by this vector store.

        Returns:
            The Embeddings instance used to generate vector representations.
        """
        return self.embedding

    @override
    def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
        """Deletes vector objects from the Ontology by their IDs.

        Executes delete actions for each provided ID, removing the corresponding
        vector objects from the Ontology object set.

        Args:
            ids: Sequence of primary keys identifying the vector objects to delete.
                If None, no deletion is performed.
            **kwargs: Additional keyword arguments (unused).
        """
        for id in ids:
            self.actions.delete(
                action_config=ActionConfig(
                    mode=ActionMode.VALIDATE_AND_EXECUTE,
                    return_edits=ReturnEditsMode.ALL,
                ),
                vector_db=id,
            )

    @override
    def add_documents(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        """Adds documents to the vector store with their embeddings.

        Generates embeddings for each document's text content and creates
        corresponding vector objects in the Ontology. Each document's metadata
        is stored as a JSON string. Only successfully validated objects are
        included in the returned IDs.

        Args:
            documents: List of Document objects to add to the vector store.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            List of primary keys (IDs) for the successfully created vector objects.
        """
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        ids = list()
        for document, vector in zip(documents, vectors):
            response = self.actions.create(
                action_config=ActionConfig(
                    mode=ActionMode.VALIDATE_AND_EXECUTE,
                    return_edits=ReturnEditsMode.ALL,
                ),
                vector=vector,
                text=document.page_content,
                metadata=json.dumps(document.metadata),
            )
            if response.validation.result == "VALID":
                # If ReturnEditsMode.ALL is used, new and updated objects edits will contain the primary key of the object
                if response.edits.type == "edits":
                    ids.append(response.edits.edits[0].primary_key)

        return ids

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        vector_store: OntologyObjectSet,
        vector_property: VectorObjectTypeProperty,
        actions: OntologyActions,
        **kwargs,
    ):
        """Creates a vector store instance and populates it with texts.

        Factory method that instantiates a PalantirOntology vector store and
        adds the provided texts as documents with optional metadata.

        Args:
            texts: List of text strings to add to the vector store.
            embedding: Embeddings model for generating vector representations.
            metadatas: Optional list of metadata dictionaries, one per text.
                Must be same length as texts if provided.
            vector_store: Ontology object set for storing vector objects.
            vector_property: Vector property type for similarity search.
            actions: Container with create and delete action types.
            **kwargs: Additional keyword arguments passed to add_texts.

        Returns:
            A new PalantirOntology instance populated with the provided texts.
        """
        store = cls(vector_store, vector_property, actions, embedding)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Retrieves documents from the vector store by their IDs.

        Fetches vector objects from the Ontology using their primary keys and
        converts them back into Document objects with text and metadata.

        Args:
            ids: Sequence of primary keys identifying the vector objects to retrieve.

        Returns:
            List of Document objects corresponding to the provided IDs, with
            page_content containing the text and metadata deserialized from JSON.
        """
        documents = list()

        for id in ids:
            response = self.vector_store.get(id)
            documents.append(
                Document(
                    page_content=response.text, metadata=json.loads(response.metadata)
                )
            )

        return documents

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Performs similarity search to find documents similar to the query.

        Embeds the query text and uses nearest neighbor search on the vector
        property to find the k most similar documents in the Ontology.
        Results are ordered by relevance with the most similar documents first.

        Args:
            query: The text query to search for similar documents.
            k: Number of most similar documents to return. Defaults to 4.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            List of up to k Document objects most similar to the query,
            ordered by relevance (most similar first).
        """
        vector_query = self.embedding.embed_query(query)
        nearest_neighbors = (
            self.vector_store.nearest_neighbors(
                query=vector_query,
                num_neighbors=k,
                vector_property=self.vector_property,
            )
            .order_by("relevance")
            .iterate()
        )

        return [
            Document(page_content=neighbor.text, metadata=json.loads(neighbor.metadata))
            for neighbor in nearest_neighbors
        ]

