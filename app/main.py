import os
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Dict
import PyPDF2
import uvicorn
from dotenv import load_dotenv
import nest_asyncio
import io
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding_model = OpenAIEmbeddings()

nest_asyncio.apply()

# Set up API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
ragflow_url = "http://localhost"  

app = FastAPI()

# Define a Pydantic model for the request body
class DeleteDatasetsRequest(BaseModel):
    ids: List[str]

class DeleteDocumentsRequest(BaseModel):
    ids: List[str] 

class ParseDocumentsRequest(BaseModel):
    document_ids: List[str]

class CreateChatSessionRequest(BaseModel):
    name: str  # Required
    user_id: Optional[str] = None 

class ParserConfig(BaseModel):
    chunk_token_num: Optional[int] = 128
    layout_recognize: Optional[bool] = True
    html4excel: Optional[bool] = False
    delimiter: Optional[str] = "\\n!?;。；！？"
    task_page_size: Optional[int] = 12
    raptor: Optional[Dict[str, bool]] = {"use_raptor": False}
    
    entity_types: Optional[List[str]] = ["organization", "person", "location", "event", "time"]

class CreateDatasetRequest(BaseModel):
    name: str  # Required
    avatar: Optional[str] = None  # Base64 encoded avatar (Optional)
    description: Optional[str] = None  # Dataset description (Optional)
    language: Optional[str] = "English"  # Default to English
    embedding_model: Optional[str] = "text-embedding-ada-002"  # Default model
    permission: Optional[str] = "me"  # Default permission (me)
    chunk_method: Optional[str] = "naive"  # Default chunk method
    parser_config: Optional[ParserConfig] = None  # Configuration for chunking

class RetrievalRequest(BaseModel):
    question: str
    dataset_ids: List[str]
    document_ids: List[str]
    page: Optional[int] = 1
    page_size: Optional[int] = 30
    similarity_threshold: Optional[float] = 0.2
    vector_similarity_weight: Optional[float] = 0.3
    top_k: Optional[int] = 1024
    rerank_id: Optional[str] = None
    keyword: Optional[bool] = False
    highlight: Optional[bool] = False

class LLMConfig(BaseModel):
    model_name: Optional[str] = "gpt-4o"
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.3
    presence_penalty: Optional[float] = 0.4
    frequency_penalty: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class PromptConfig(BaseModel):
    similarity_threshold: Optional[float] = 0.2
    keywords_similarity_weight: Optional[float] = 0.3
    top_n: Optional[int] = 8
    variables: Optional[List[Dict[str, Any]]] = [{"key": "knowledge", "optional": True}]
    rerank_model: Optional[str] = None
    top_k: Optional[int] = 1024
    empty_response: Optional[str] = "Sorry! No relevant content was found in the knowledge base!"
    opener: Optional[str] = "Hi! I am your assistant, can I help you?"
    show_quote: Optional[bool] = True
    prompt: Optional[str] = "You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question."

class CreateChatAssistantRequest(BaseModel):
    name: str  # Required
    avatar: Optional[str] = None  # Base64 encoded avatar (Optional)
    dataset_ids: List[str]  # List of dataset IDs associated with the assistant
    llm: Optional[LLMConfig] = None  # LLM settings for the chat assistant
    prompt: Optional[PromptConfig] = None  # Instructions for the assistant

# Create Dataset
@app.post("/create-dataset")
async def create_dataset(data: CreateDatasetRequest):
    response = requests.post(
        f"{ragflow_url}/api/v1/datasets",
        json=data.model_dump(),
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to create dataset")

    return response.json()


@app.post("/upload/{dataset_id}")
async def upload_file(dataset_id: str, file: UploadFile = File(...)):
    # Read the file content
    file_content = await file.read()

    # Check if the file is PDF or TXT
    if file.filename.endswith('.pdf'):
        file_stream = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file.filename.endswith('.txt'):
        text = file_content.decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Upload document in the specified dataset
    document_response = requests.post(
        f"{ragflow_url}/api/v1/datasets/{dataset_id}/documents",
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"},
        files={"file": (file.filename, file_content, file.content_type)}
    )

    if document_response.status_code != 200:
        raise HTTPException(status_code=document_response.status_code, detail="Failed to upload document")

    document_data = document_response.json().get("data", [{}])[0]
    if not document_data:
        raise HTTPException(status_code=500, detail="Failed to retrieve document data")

    # Prepare the response in the expected format, including dataset_id
    success_response = {
        "code": 0,
        "data": [
            {
                "chunk_method": "naive",
                "created_by": document_data.get("created_by", ""),
                "dataset_id": dataset_id, 
                "id": document_data.get("id"),
                "location": document_data.get("location"),
                "name": document_data.get("name"),
                "parser_config": document_data.get("parser_config", {}),
                "run": document_data.get("run", "UNSTART"),
                "size": document_data.get("size"),
                "thumbnail": document_data.get("thumbnail", ""),
                "type": document_data.get("type")
            }
        ]
    }

    return success_response


@app.get("/list-datasets")
async def list_datasets(page: int = 1, page_size: int = 30, orderby: str = "create_time", desc: bool = True, name: str = "", dataset_id: str = ""):
    response = requests.get(
        f"{ragflow_url}/api/v1/datasets",
        params={
            "page": page,
            "page_size": page_size,
            "orderby": orderby,
            "desc": str(desc).lower(),  
            "name": name,
            "id": dataset_id
        },
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve datasets")

    dataset_data = response.json()

    # Check if datasets are available
    if "data" not in dataset_data or not dataset_data["data"]:
        raise HTTPException(status_code=404, detail="No datasets found")

    return dataset_data["data"]


@app.delete("/delete-datasets")
async def delete_datasets(data: DeleteDatasetsRequest):
    dataset_ids = data.get("ids")

    if not dataset_ids:
        raise HTTPException(status_code=400, detail="No dataset IDs provided")

    response = requests.delete(
        f"{ragflow_url}/api/v1/datasets",
        json={"ids": dataset_ids},
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to delete datasets")

    return {"code": 0, "message": "Datasets deleted successfully"}


# List Documents from Dataset
@app.get("/list-documents")
async def list_documents(dataset_id: str, page: int = 1, page_size: int = 30, orderby: str = "create_time", desc: bool = True, keywords: str = "", document_id: str = "", document_name: str = ""):
    response = requests.get(
        f"{ragflow_url}/api/v1/datasets/{dataset_id}/documents",
        params={
            "page": page,
            "page_size": page_size,
            "orderby": orderby,
            "desc": desc,
            "keywords": keywords,
            "id": document_id,
            "name": document_name
        },
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve documents")

    document_data = response.json()

    # Check if documents are available
    if "data" not in document_data or "docs" not in document_data["data"]:
        raise HTTPException(status_code=404, detail="No documents found")

    return document_data["data"]

# Delete Document from Dataset
@app.delete("/delete-documents/{dataset_id}")
async def delete_documents(dataset_id: str, data: DeleteDocumentsRequest):
    # Check if "ids" is provided in the body
    document_ids = data.get("ids", [])
    
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided for deletion")
    
    payload = {
        "ids": document_ids
    }
    
    response = requests.delete(
        f"{ragflow_url}/api/v1/datasets/{dataset_id}/documents",
        json=payload,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )
    
    if response.status_code != 200:
        error_data = response.json()
        if "message" in error_data:
            raise HTTPException(status_code=400, detail=error_data["message"])
        raise HTTPException(status_code=response.status_code, detail="Failed to delete documents")

    return {"code": 0, "message": "Documents deleted successfully"}

# Define the endpoint to parse documents in a dataset
@app.post("/parse-documents/{dataset_id}")
async def parse_documents(dataset_id: str, data: ParseDocumentsRequest):
    if not data.document_ids:
        raise HTTPException(status_code=400, detail="`document_ids` is required")

    payload = data.model_dump()

    response = requests.post(
        f"{ragflow_url}/api/v1/datasets/{dataset_id}/chunks",
        json=payload,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to parse documents")

    return response.json()


@app.get("/list-chunks/{dataset_id}/{document_id}")
async def list_chunks(
    dataset_id: str,
    document_id: str,
    keywords: Optional[str] = None,
    page: Optional[int] = 1,
    page_size: Optional[int] = 1024,
    chunk_id: Optional[str] = None
):
    params = {
        "keywords": keywords,
        "page": page,
        "page_size": page_size,
        "id": chunk_id
    }

    # Clean up any None values from the parameters to avoid sending unwanted query params
    params = {key: value for key, value in params.items() if value is not None}

    response = requests.get(
        f"{ragflow_url}/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
        params=params,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve chunks")

    return response.json()


# Define the endpoint to retrieve chunks
@app.post("/retrieve-chunks")
async def retrieve_chunks(data: RetrievalRequest):
    payload = data.model_dump()

    response = requests.post(
        f"{ragflow_url}/api/v1/retrieval",
        json=payload,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve chunks")

    return response.json()

@app.post("/create-chat-assistant")
async def create_chat_assistant(data: CreateChatAssistantRequest):
    llm_config = data.llm or LLMConfig()
    prompt_config = data.prompt or PromptConfig()

    if prompt_config.rerank_model is None:
        prompt_config.rerank_model = ""

    payload = {
        "name": data.name,
        "avatar": data.avatar,
        "dataset_ids": data.dataset_ids,
        "llm": llm_config.model_dump(),
        "prompt": prompt_config.model_dump()
    }

    response = requests.post(
        f"{ragflow_url}/api/v1/chats",
        json=payload,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
    )

    if response.status_code != 200:
        error_message = response.json().get("message", "Failed to create chat assistant")
        raise HTTPException(status_code=response.status_code, detail=error_message)

    return response.json()

# Define the endpoint to list chat assistants
@app.get("/list-chat-assistants")
async def list_chats(page: int = 1, page_size: int = 30, orderby: str = "create_time", desc: bool = True, name: str = "", chat_id: str = ""):
    response = requests.get(
        f"{ragflow_url}/api/v1/chats",
        params={
            "page": page,
            "page_size": page_size,
            "orderby": orderby,
            "desc": str(desc).lower(),  
            "name": name,
            "id": chat_id
        },
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve chat assistants")

    chat_data = response.json()

    # Check if chat assistants are available
    if "data" not in chat_data or not chat_data["data"]:
        raise HTTPException(status_code=404, detail="No chat assistants found")

    return chat_data["data"]

@app.delete("/delete-chat-assistants")
async def delete_chat_assistants(data: DeleteDatasetsRequest):
    chat_ids = data.ids  # Extract the list of chat assistant IDs from the request body

    if not chat_ids:
        raise HTTPException(status_code=400, detail="No chat assistant IDs provided")

    response = requests.delete(
        f"{ragflow_url}/api/v1/chats",
        json={"ids": chat_ids},
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to delete chat assistants")

    return {"code": 0, "message": "Chat assistants deleted successfully"}

# Define the endpoint to create a chat session
@app.post("/create-chat-session/{chat_id}")
async def create_chat_session(chat_id: str, data: CreateChatSessionRequest):
    session_name = data.name
    user_id = data.user_id  # Optional

    if not session_name:
        raise HTTPException(status_code=400, detail="Session name is required")

    payload = {
        "name": session_name,
        "user_id": user_id
    }

    response = requests.post(
        f"{ragflow_url}/api/v1/chats/{chat_id}/sessions",
        json=payload,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
    )

    if response.status_code != 200:
        error_message = response.json().get("message", "Failed to create chat session")
        raise HTTPException(status_code=response.status_code, detail=error_message)

    return response.json()

@app.get("/list-chat-sessions/{chat_id}")
async def list_chat_sessions(
    chat_id: str,
    page: int = 1,
    page_size: int = 30,
    orderby: str = "create_time",
    desc: bool = True,
    name: str = "",
    session_id: str = "",
    user_id: str = ""
):
    params = {
        "page": page,
        "page_size": page_size,
        "orderby": orderby,
        "desc": str(desc).lower(),  
        "name": name,
        "id": session_id,
        "user_id": user_id
    }

    response = requests.get(
        f"{ragflow_url}/api/v1/chats/{chat_id}/sessions",
        params=params,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to retrieve chat sessions")

    session_data = response.json()

    # Check if sessions are available
    if "data" not in session_data or not session_data["data"]:
        raise HTTPException(status_code=404, detail="No sessions found")

    return session_data["data"]

# Endpoint to delete chat assistant's sessions
@app.delete("/delete-chat-sessions/{chat_id}")
async def delete_chat_sessions(chat_id: str, data: DeleteDatasetsRequest):
    session_ids = data.ids

    if not session_ids:
        raise HTTPException(status_code=400, detail="No session IDs provided")

    payload = {
        "ids": session_ids
    }

    response = requests.delete(
        f"{ragflow_url}/api/v1/chats/{chat_id}/sessions",
        json=payload,
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
    )

    if response.status_code != 200:
        error_message = response.json().get("message", "Failed to delete chat sessions")
        raise HTTPException(status_code=response.status_code, detail=error_message)

    return response.json()

# Function to evaluate model responses using multiple metrics
def evaluate_response(reference_list, generated):
    """Evaluates the response using BLEU, ROUGE, METEOR, Exact Match, and Ragas metrics."""
    ref_tokens = reference_list.lower().split() 
    gen_tokens = generated.lower().split()

    # BLEU Score
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
    
    # ROUGE Score
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference_list, generated) 

    # METEOR Score
    meteor = meteor_score([ref_tokens], gen_tokens)

    # Exact Match (EM)
    exact_match = 1.0 if reference_list.strip().lower() == generated.strip().lower() else 0.0    
    return bleu, rouge_scores, meteor, exact_match

@app.post("/chat/{chat_id}/completions")
async def chat_with_assistant(
    chat_id: str,
    question: str,
    stream: bool = False,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):

    if not question:
        raise HTTPException(status_code=400, detail="Please input your question.")
    
    payload = {
        "question": question,
        "stream": stream,
        "session_id": session_id,
        "user_id": user_id
    }

    # Track the time when the request is sent
    start_time = time.time()

    try:
        response = requests.post(
            f"{ragflow_url}/api/v1/chats/{chat_id}/completions",
            json=payload,
            headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"}
        ) 
        # Calculate response time
        response_time = time.time() - start_time
  
        if response.status_code == 200:
            data = response.json()  
            bot_response = data["data"]["answer"]
            reference_chunks = data["data"]["reference"]["chunks"]
            reference_responses = " ".join(chunk["content"] for chunk in reference_chunks)

            bleu_score, rouge_score, meteor_score, exact_match_score = evaluate_response(reference_responses, bot_response)

            print("Evaluation Results:")
            print(f"Response Time: {response_time} seconds")
            print(f"BLEU Score: {bleu_score:.4f}")
            print(f"ROUGE Score: {rouge_score}")
            print(f"METEOR Score: {meteor_score:.4f}")
            print(f"Exact Match (EM) Score: {exact_match_score:.2f}%")

            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get completions from the chat assistant.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Define the new endpoint to create a session with the agent
@app.post("/agents/{agent_id}/sessions")
async def create_chat_session(agent_id: str, user_id: Optional[str] = None):
    api_url = f"{ragflow_url}/api/v1/agents/{agent_id}/sessions"
    
    params = {}
    if user_id:
        params["user_id"] = user_id
    
    response = requests.post(
        api_url,
        params=params,  
        headers={"Authorization": f"Bearer {RAGFLOW_API_KEY}", "Content-Type": "application/json"},
    )
    
    if response.status_code != 200:
        error_data = response.json()
        raise HTTPException(status_code=response.status_code, detail=error_data.get("message", "Failed to create session"))
    
    return response.json()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
