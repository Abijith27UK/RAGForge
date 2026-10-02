"""Indexing services: chunk -> embed -> index, incrementally."""
from app.services.indexing.document_indexer import (
    IndexingError,
    IndexOutcome,
    chunks_of_document,
    count_chunks_for_documents,
    index_documents,
    read_parsed_text,
)

__all__ = [
    "IndexingError",
    "IndexOutcome",
    "chunks_of_document",
    "count_chunks_for_documents",
    "index_documents",
    "read_parsed_text",
]