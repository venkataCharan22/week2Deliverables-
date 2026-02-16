"""
RAG Chain Implementation
Combines retrieval and generation for question answering
"""

from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGChain:
    """RAG Chain for question answering over documents"""

    def __init__(self, vector_db_manager, model_name: str = "llama3", temperature: float = 0.7):
        """
        Initialize RAG Chain

        Args:
            vector_db_manager: VectorDBManager instance
            model_name: Ollama model name
            temperature: LLM temperature
        """
        self.vector_db = vector_db_manager
        self.model_name = model_name
        self.temperature = temperature

        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=temperature
        )

        # Initialize retriever
        self.retriever = None
        self._setup_retriever()

        # Initialize chain
        self.chain = None
        self._setup_chain()

    def _setup_retriever(self, k: int = 4):
        """Set up the retriever"""
        self.retriever = self.vector_db.get_retriever(k=k)

    def _setup_chain(self):
        """Set up the RAG chain using LCEL"""

        # Create prompt template
        template = """You are a helpful AI assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.
Provide a detailed and informative answer when possible.

Context:
{context}

Question: {question}

Answer: """

        prompt = ChatPromptTemplate.from_template(template)

        # Create LCEL chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs) -> str:
        """Format retrieved documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> Dict:
        """
        Ask a question using RAG

        Args:
            question: User question

        Returns:
            Dictionary containing answer and sources
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)

            # Generate answer
            answer = self.chain.invoke(question)

            # Format sources
            sources = []
            for i, doc in enumerate(retrieved_docs):
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "chunk_index": i + 1
                })

            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }

        except Exception as e:
            return {
                "status": "error",
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "num_sources": 0
            }

    def ask_with_scores(self, question: str, k: int = 4) -> Dict:
        """
        Ask a question and include similarity scores

        Args:
            question: User question
            k: Number of documents to retrieve

        Returns:
            Dictionary containing answer, sources, and scores
        """
        try:
            # Perform similarity search with scores
            results = self.vector_db.similarity_search(question, k=k)

            if not results:
                return {
                    "status": "error",
                    "question": question,
                    "answer": "No relevant documents found in the database.",
                    "sources": [],
                    "scores": []
                }

            # Format context for the LLM
            context = "\n\n".join([content for content, score, metadata in results])

            # Create prompt
            prompt_text = f"""You are a helpful AI assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.
Provide a detailed and informative answer when possible.

Context:
{context}

Question: {question}

Answer: """

            # Generate answer
            answer = self.llm.invoke(prompt_text)

            # Extract answer text
            if hasattr(answer, 'content'):
                answer_text = answer.content
            else:
                answer_text = str(answer)

            # Format sources with scores
            sources = []
            scores = []
            for i, (content, score, metadata) in enumerate(results):
                sources.append({
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "full_content": content,
                    "metadata": metadata,
                    "chunk_index": i + 1,
                    "similarity_score": round(score, 4)
                })
                scores.append(round(score, 4))

            return {
                "status": "success",
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "scores": scores,
                "num_sources": len(sources)
            }

        except Exception as e:
            return {
                "status": "error",
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "scores": []
            }

    def update_model(self, model_name: str, temperature: float = None):
        """
        Update the LLM model

        Args:
            model_name: New model name
            temperature: New temperature (optional)
        """
        self.model_name = model_name

        if temperature is not None:
            self.temperature = temperature

        self.llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=self.temperature
        )

        # Rebuild chain with new LLM
        self._setup_chain()

    def update_retriever_k(self, k: int):
        """
        Update the number of documents to retrieve

        Args:
            k: Number of documents
        """
        self._setup_retriever(k=k)
        self._setup_chain()
