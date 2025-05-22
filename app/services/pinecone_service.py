# app/services/pinecone_service.py
from pinecone import Pinecone
from typing import List, Dict, Any, Tuple
from app.config import get_settings
from app.services.openai_service import OpenAIService
from app.models.schemas import Reference

settings = get_settings()

class PineconeService:
    def __init__(self):
        """
        Initialize Pinecone service with the configured settings.
        """
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            # Get the list of indexes
            self.indexes = self.pc.list_indexes()
            # Initialize OpenAI service for embeddings
            self.openai_service = OpenAIService()
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone service: {str(e)}")

    def get_index(self, index_name: str):
        """
        Get a Pinecone index by name.
        """
        try:
            return self.pc.Index(index_name)
        except Exception as e:
            raise Exception(f"Failed to get index {index_name}: {str(e)}")

    async def query_documents(self, query: str, index_name: str, namespace: str = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Query documents using the provided query string.

        Args:
            query (str): The query string to search for
            index_name (str): The name of the index to query
            namespace (str, optional): The namespace to search in
            top_k (int): Number of results to return

        Returns:
            Dict containing the query results
        """
        try:
            # Get the index
            index = self.get_index(index_name)

            # Generate embedding vector for the query
            query_vector = await self.openai_service.generate_embedding(query)

            # Perform the query with cosine similarity (default in Pinecone)
            query_response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )

            return query_response

        except Exception as e:
            raise Exception(f"Pinecone query failed: {str(e)}")

    def extract_context_and_references(self, query_response: Dict[str, Any]) -> Tuple[str, List[Reference]]:
        """
        Extract context and references from the query response.

        Args:
            query_response (Dict): The response from Pinecone query

        Returns:
            Tuple containing the combined context string and list of references
        """
        try:
            contexts = []
            references = []

            # Extract text and references from matches
            for match in query_response.matches:
                if match.metadata:
                    # Extract text if available
                    if 'text' in match.metadata:
                        contexts.append(match.metadata['text'])

                    # Extract file name and page if available
                    if 'file_name' in match.metadata and 'page' in match.metadata:
                        references.append(Reference(
                            file_name=match.metadata['file_name'],
                            page=match.metadata['page']
                        ))

            # Combine all context texts with newlines
            combined_context = "\n\n".join(contexts) if contexts else ""

            return combined_context, references

        except Exception as e:
            raise Exception(f"Failed to extract context: {str(e)}")

    async def search_and_process_query(self, query: str, index_name: str, namespace: str = None, top_k: int = 5) -> Tuple[str, List[Reference]]:
        """
        Perform a complete search operation: query documents and extract context.

        Args:
            query (str): The search query
            index_name (str): The name of the index to query
            namespace (str, optional): The namespace to search in
            top_k (int): Number of results to return

        Returns:
            Tuple containing the context and references
        """
        try:
            # Query the documents
            query_response = await self.query_documents(query, index_name, namespace, top_k)

            # Extract and return context and references
            return self.extract_context_and_references(query_response)

        except Exception as e:
            raise Exception(f"Search and process operation failed: {str(e)}")

    def list_namespaces(self, index_name: str) -> List[str]:
        """
        List all namespaces in the given index.

        Args:
            index_name (str): The name of the index

        Returns:
            List of namespace names
        """
        try:
            index = self.get_index(index_name)
            stats = index.describe_index_stats()
            return list(stats.namespaces.keys())
        except Exception as e:
            raise Exception(f"Failed to list namespaces: {str(e)}")
