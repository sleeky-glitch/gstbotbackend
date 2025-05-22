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
            # New Pinecone client initialization (v3.x+)
            self.pc = pinecone.Pinecone(api_key=settings.pinecone_api_key)
            self.openai_service = OpenAIService()
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone service: {str(e)}")

    def list_indexes(self) -> List[str]:
        try:
            return [idx.name for idx in self.pc.list_indexes().indexes]
        except Exception as e:
            raise Exception(f"Failed to list indexes: {str(e)}")

    def get_index(self, index_name: str):
        try:
            return self.pc.Index(index_name)
        except Exception as e:
            raise Exception(f"Failed to get index {index_name}: {str(e)}")

    async def query_documents(self, query: str, index_name: str, namespace: str = None, top_k: int = 5) -> Dict[str, Any]:
        try:
            index = self.get_index(index_name)
            query_vector = await self.openai_service.generate_embedding(query)
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
        try:
            contexts = []
            references = []
            for match in query_response['matches']:
                if 'metadata' in match:
                    if 'text' in match['metadata']:
                        contexts.append(match['metadata']['text'])
                    if 'file_name' in match['metadata'] and 'page' in match['metadata']:
                        references.append(Reference(
                            file_name=match['metadata']['file_name'],
                            page=match['metadata']['page']
                        ))
            combined_context = "\n\n".join(contexts) if contexts else ""
            return combined_context, references
        except Exception as e:
            raise Exception(f"Failed to extract context: {str(e)}")

    async def search_and_process_query(self, query: str, index_name: str, namespace: str = None, top_k: int = 5) -> Tuple[str, List[Reference]]:
        try:
            query_response = await self.query_documents(query, index_name, namespace, top_k)
            return self.extract_context_and_references(query_response)
        except Exception as e:
            raise Exception(f"Search and process operation failed: {str(e)}")

    def list_namespaces(self, index_name: str) -> List[str]:
        try:
            index = self.get_index(index_name)
            stats = index.describe_index_stats()
            return list(stats['namespaces'].keys())
        except Exception as e:
            raise Exception(f"Failed to list namespaces: {str(e)}")
