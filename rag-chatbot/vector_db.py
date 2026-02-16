"""
Vector Database Manager for RAG System
Handles ChromaDB operations including embedding generation and similarity search
"""

from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


class VectorDBManager:
    """Manage ChromaDB operations for RAG system"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize Vector Database Manager

        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the Chroma vector store"""
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            raise Exception(f"Error initializing vector store: {str(e)}")

    def add_documents(self, chunks: List, metadata: List[Dict] = None) -> Dict:
        """
        Add documents to the vector database

        Args:
            chunks: List of document chunks
            metadata: Optional metadata for each chunk

        Returns:
            Dictionary with operation results
        """
        try:
            if not chunks:
                return {"status": "error", "message": "No chunks provided"}

            # Extract text content from chunks
            texts = [chunk.page_content for chunk in chunks]

            # Add metadata from chunks if not provided
            if metadata is None:
                metadata = [chunk.metadata for chunk in chunks]

            # Add to vector store
            ids = self.vector_store.add_texts(texts=texts, metadatas=metadata)

            return {
                "status": "success",
                "num_documents": len(ids),
                "message": f"Successfully added {len(ids)} documents to vector database"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error adding documents: {str(e)}"
            }

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict]]:
        """
        Perform similarity search

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of tuples (content, score, metadata)
        """
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # Filter by score threshold and format results
            filtered_results = []
            for doc, score in results:
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1 - score if score <= 1 else 1 / (1 + score)

                if similarity >= score_threshold:
                    filtered_results.append((
                        doc.page_content,
                        similarity,
                        doc.metadata
                    ))

            return filtered_results

        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")

    def get_retriever(self, k: int = 4):
        """
        Get a retriever for the vector store

        Args:
            k: Number of documents to retrieve

        Returns:
            LangChain retriever object
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get the underlying ChromaDB collection
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(name=self.collection_name)

            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "embedding_dimension": 384,  # all-MiniLM-L6-v2 dimension
                "status": "active" if count > 0 else "empty"
            }

        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "embedding_dimension": 384,
                "status": "not_initialized",
                "error": str(e)
            }

    def delete_collection(self) -> Dict:
        """
        Delete the entire collection

        Returns:
            Dictionary with operation result
        """
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=self.collection_name)

            # Reinitialize vector store
            self._initialize_vector_store()

            return {
                "status": "success",
                "message": f"Collection '{self.collection_name}' deleted successfully"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting collection: {str(e)}"
            }

    def search_by_metadata(self, metadata_filter: Dict, k: int = 4) -> List:
        """
        Search documents by metadata filter

        Args:
            metadata_filter: Dictionary of metadata filters
            k: Number of results to return

        Returns:
            List of matching documents
        """
        try:
            # This is a basic implementation
            # ChromaDB supports more advanced filtering
            results = self.vector_store.similarity_search(
                query="",  # Empty query for metadata-only search
                k=k,
                filter=metadata_filter
            )
            return results

        except Exception as e:
            raise Exception(f"Error searching by metadata: {str(e)}")
