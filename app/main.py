# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import QueryRequest, QueryResponse
from app.services.pinecone_service import PineconeService
from app.services.openai_service import OpenAIService

app = FastAPI(title="PDF Query API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pinecone_service = PineconeService()
openai_service = OpenAIService()

@app.get("/")
async def root():
    return {"message": "PDF Query API is running"}

@app.get("/namespaces/{index_name}")
async def list_namespaces(index_name: str):
    """List all namespaces in the specified index."""
    try:
        namespaces = pinecone_service.list_namespaces(index_name)
        return {"namespaces": namespaces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest, index_name: str):
    """
    Query the PDF database across all namespaces and get a response.

    - **request**: The query request containing the query text and top_k
    - **index_name**: The name of the Pinecone index to query
    """
    print(f"Received request: {request}, index: {index_name}")
    try:
        # Query Pinecone for relevant context across all namespaces
        context, references = await pinecone_service.search_across_all_namespaces(
            request.query,
            index_name,
            request.top_k
        )

        # If no context found, return early
        if not context:
            return QueryResponse(
                response="I don't have enough information to answer this question.",
                success=True,
                references=[]
            )

        # Generate response using OpenAI
        response_text = await openai_service.generate_response(context, request.query)

        return QueryResponse(
            response=response_text,
            success=True,
            references=references
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
