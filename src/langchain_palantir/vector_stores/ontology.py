from typing_extensions import override, Any, Sequence
import json
from langchain_core.vectorstores.base import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from foundry_sdk_runtime.types import (
    ActionConfig,
    ActionMode,
    ReturnEditsMode,
)
from foundry_sdk_runtime.ontology_object_set import OntologyObjectSet
from foundry_sdk_runtime.properties.property_types import VectorObjectTypeProperty


class OntologyActions:
    def __init__(self, create, delete) -> None:
        self.create = create
        self.delete = delete


class PalantirOntology(VectorStore):
    def __init__(
        self,
        vector_store: OntologyObjectSet,
        vector_property: VectorObjectTypeProperty,
        actions: OntologyActions,
        embedding: Embeddings,
    ) -> None:
        self.vector_store = vector_store
        self.vector_property = vector_property
        self.actions = actions
        self.embedding = embedding

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding

    @override
    def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
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
        store = cls(vector_store, vector_property, actions, embedding)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
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

        print(type(nearest_neighbors))

        return [
            Document(page_content=neighbor.text, metadata=json.loads(neighbor.metadata))
            for neighbor in nearest_neighbors
        ]
