"""
Base class for document chunkers.
"""

from abc import ABC, abstractmethod
from typing import List
from evidencebench.models import Document, Chunk


class BaseChunker(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
