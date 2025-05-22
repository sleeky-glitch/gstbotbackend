# app/utils/prompts.py
SYSTEM_PROMPT = """You are a helpful assistant that provides accurate information based on the context provided.
When answering questions:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Cite specific sections, pages, or documents when possible
4. Format your responses clearly with proper punctuation and structure
5. Break down complex information into digestible parts
6. Be concise but thorough"""

def get_chat_prompt(context: str, user_input: str) -> str:
    return f"""Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the following query:
{user_input}

If the context doesn't contain the answer, just say "I don't have enough information to answer this question."
Include references to specific documents and page numbers when possible."""
