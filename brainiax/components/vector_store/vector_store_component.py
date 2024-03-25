import logging
import typing

from injector import inject, singleton
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorStoreIndex
from llama_index.core.vector_stores.types import (
   FilterCondition,
   MetadataFilter,
   MetadataFilters,
   VectorStore,
)

from brainiax.open_ai.extensions.context_filter import ContextFilter
from brainiax.paths import local_data_path
from brainiax.settings.settings import Settings

logger = logging.getLogger(__name__)

def _doc_id_metadata_filter(
   context_filter: ContextFilter | None,
) -> MetadataFilters:
   """
   Creates metadata filters for restricting retrieval to specific document IDs.
   """

   filters = MetadataFilters(filters=[], condition=FilterCondition.OR)

   if context_filter and context_filter.docs_ids:
       for doc_id in context_filter.docs_ids:
           filters.filters.append(MetadataFilter(key="doc_id", value=doc_id))

   return filters


@singleton
class VectorStoreComponent:
   """
   This class manages the connection and interaction with the vector store.

   Attributes:
       settings (Settings): The application settings object.
       vector_store (VectorStore): The loaded vector store instance.
   """

   settings: Settings
   vector_store: VectorStore

   @inject
   def __init__(self, settings: Settings) -> None:
       """
       Initializes the vector store based on the provided settings.
       """

       self.settings = settings

       try:
           from llama_index.vector_stores.qdrant import QdrantVectorStore  # type: ignore
           from qdrant_client import QdrantClient  # type: ignore
       except ImportError as e:
           raise ImportError(
               "Qdrant dependencies not found, install with `poetry install --extras vector-stores-qdrant`"
           ) from e

       if settings.qdrant is None:
           logger.info(
               "Qdrant config not found. Using default settings. "
               "Trying to connect to Qdrant at localhost:6333."
           )
           client = QdrantClient()
       else:
           client = QdrantClient(**settings.qdrant.model_dump(exclude_none=True))

       self.vector_store = typing.cast(
           VectorStore,
           QdrantVectorStore(
               client=client,
               collection_name="make_this_parameterizable_per_api_call",  # TODO
           ),
       )

   def get_retriever(
       self,
       index: VectorStoreIndex,
       context_filter: ContextFilter | None = None,
       similarity_top_k: int = 2,
   ) -> VectorIndexRetriever:
       """
       Creates a retriever for the given index, handling potential filtering for Qdrant and other vector stores.
       """

       return VectorIndexRetriever(
           index=index,
           similarity_top_k=similarity_top_k,
           doc_ids=context_filter.docs_ids if context_filter else None,
           filters=(
               _doc_id_metadata_filter(context_filter)
               if self.settings.vectorstore.database != "qdrant"
               else None
           ),
       )

   def close(self) -> None:
       """
       Closes the vector store client if it has a close method.
       """

       if hasattr(self.vector_store.client, "close"):
           self.vector_store.client.close()
