"""
Document Processor for RAG System
Handles document loading, text extraction, and chunking
"""

from typing import List, Dict
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


class DocumentProcessor:
    """Process documents for RAG system"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Dict]:
        """
        Load a document from file

        Args:
            file_path: Path to the document file

        Returns:
            List of document chunks with metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            return documents

        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")

    def chunk_document(self, documents: List) -> List[Dict]:
        """
        Split documents into chunks

        Args:
            documents: List of LangChain documents

        Returns:
            List of document chunks
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            raise Exception(f"Error chunking document: {str(e)}")

    def process_document(self, file_path: str) -> Dict:
        """
        Complete document processing pipeline

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing chunks and metadata
        """
        try:
            # Load document
            documents = self.load_document(file_path)

            # Chunk document
            chunks = self.chunk_document(documents)

            # Extract metadata
            filename = os.path.basename(file_path)

            return {
                "filename": filename,
                "file_path": file_path,
                "num_chunks": len(chunks),
                "chunks": chunks,
                "total_chars": sum(len(chunk.page_content) for chunk in chunks)
            }

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def get_chunk_stats(self, chunks: List) -> Dict:
        """
        Get statistics about chunks

        Args:
            chunks: List of document chunks

        Returns:
            Dictionary of statistics
        """
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_lengths) / len(chunk_lengths) if chunks else 0,
            "min_chunk_size": min(chunk_lengths) if chunks else 0,
            "max_chunk_size": max(chunk_lengths) if chunks else 0,
            "total_characters": sum(chunk_lengths)
        }
