# app/services/pinecone_service.py
import pinecone
from typing import List, Dict, Any, Tuple
from app.config import get_settings
from app.services.openai_service import OpenAIService
from app.models.schemas import Reference

settings = get_settings()

class PineconeService:
    def __init__(self):
        try:
            # Initialize Pinecone client
            self.pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
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

    def list_namespaces(self, index_name: str) -> List[str]:
        """
        List all namespaces in the given index.
        """
        try:
            index = self.get_index(index_name)
            stats = index.describe_index_stats()
            return list(stats['namespaces'].keys())
        except Exception as e:
            raise Exception(f"Failed to list namespaces: {str(e)}")

    async def query_all_namespaces(self, query: str, index_name: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query all namespaces in the index and combine results.

        Args:
            query (str): The query string to search for
            index_name (str): The name of the index to query
            top_k (int): Number of results to return per namespace

        Returns:
            Combined results from all namespaces
        """
        try:
            index = self.get_index(index_name)
            namespaces = self.list_namespaces(index_name)

            # Generate embedding vector for the query
            query_vector = await self.openai_service.generate_embedding(query)

            all_results = []

            # Query each namespace
            for namespace in namespaces:
                try:
                    # Perform the query with cosine similarity (default in Pinecone)
                    query_response = index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace
                    )

                    # Add namespace information to each match
                    for match in query_response['matches']:
                        match['namespace'] = namespace
                        all_results.append(match)
                except Exception as e:
                    print(f"Error querying namespace {namespace}: {str(e)}")
                    continue

            # Sort all results by score (descending) and take top_k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = all_results[:top_k]

            return {"matches": top_results}

        except Exception as e:
            raise Exception(f"Pinecone query failed: {str(e)}")

    def extract_context_and_references(self, query_response: Dict[str, Any]) -> Tuple[str, List[Reference]]:
        """
        Extract context and references from the query response.
        """
        try:
            contexts = []
            references = []

            # Extract text and references from matches
            for match in query_response['matches']:
                if 'metadata' in match:
                    # Extract text if available
                    if 'text' in match['metadata']:
                        # Add namespace information to context
                        namespace_info = f"[Source: {match['namespace']}]\n"
                        contexts.append(namespace_info + match['metadata']['text'])

                    # Extract file name and page if available
                    if 'file_name' in match['metadata'] and 'page' in match['metadata']:
                        references.append(Reference(
                            file_name=match['metadata']['file_name'],
                            page=match['metadata']['page'],
                            namespace=match['namespace']
                        ))

            # Combine all context texts with newlines
            combined_context = "\n\n".join(contexts) if contexts else ""

            return combined_context, references

        except Exception as e:
            raise Exception(f"Failed to extract context: {str(e)}")

    async def search_across_all_namespaces(self, query: str, index_name: str, top_k: int = 5) -> Tuple[str, List[Reference]]:
        """
        Search across all namespaces and process the query.
        """
        try:
            # Query all namespaces
            query_response = await self.query_all_namespaces(query, index_name, top_k)

            # Extract and return context and references
            return self.extract_context_and_references(query_response)

        except Exception as e:
            raise Exception(f"Search and process operation failed: {str(e)}")
