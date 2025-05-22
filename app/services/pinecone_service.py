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

    def list_indexes(self) -> List[str]:
        """
        List all available Pinecone indexes.
        """
        try:
            return [idx.name for idx in self.pc.list_indexes().indexes]
        except Exception as e:
            raise Exception(f"Failed to list indexes: {str(e)}")

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

    async def query_namespace(self, query_vector: List[float], index_name: str, namespace: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query a specific namespace in an index.
        """
        try:
            index = self.get_index(index_name)
            query_response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )

            # Add index and namespace information to each match
            for match in query_response['matches']:
                match['index_name'] = index_name
                match['namespace'] = namespace

            return query_response['matches']
        except Exception as e:
            print(f"Error querying {index_name}/{namespace}: {str(e)}")
            return []

    async def query_entire_database(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query all indexes and all namespaces in the database.

        Args:
            query (str): The query string to search for
            top_k (int): Number of results to return per namespace

        Returns:
            Combined results from all indexes and namespaces
        """
        try:
            # Generate embedding vector for the query
            query_vector = await self.openai_service.generate_embedding(query)

            all_results = []

            # Get all indexes
            indexes = self.list_indexes()

            # Query each index and namespace
            for index_name in indexes:
                try:
                    # Get all namespaces in this index
                    namespaces = self.list_namespaces(index_name)

                    # Query each namespace
                    for namespace in namespaces:
                        matches = await self.query_namespace(query_vector, index_name, namespace, top_k)
                        all_results.extend(matches)
                except Exception as e:
                    print(f"Error processing index {index_name}: {str(e)}")
                    continue

            # Sort all results by score (descending) and take top_k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = all_results[:top_k]

            return {"matches": top_results}

        except Exception as e:
            raise Exception(f"Database-wide query failed: {str(e)}")

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
                        # Add source information to context
                        source_info = f"[Source: {match['index_name']}/{match['namespace']}]\n"
                        contexts.append(source_info + match['metadata']['text'])

                    # Extract file name and page if available
                    if 'file_name' in match['metadata'] and 'page' in match['metadata']:
                        references.append(Reference(
                            file_name=match['metadata']['file_name'],
                            page=match['metadata']['page'],
                            namespace=match['namespace'],
                            index_name=match['index_name']
                        ))

            # Combine all context texts with newlines
            combined_context = "\n\n".join(contexts) if contexts else ""

            return combined_context, references

        except Exception as e:
            raise Exception(f"Failed to extract context: {str(e)}")

    async def search_entire_database(self, query: str, top_k: int = 5) -> Tuple[str, List[Reference]]:
        """
        Search across all indexes and namespaces and process the query.
        """
        try:
            # Query the entire database
            query_response = await self.query_entire_database(query, top_k)

            # Extract and return context and references
            return self.extract_context_and_references(query_response)

        except Exception as e:
            raise Exception(f"Database-wide search failed: {str(e)}")
