import os
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# --------------------------------------------------------------------------------
# 1) Load environment variables and configure
# --------------------------------------------------------------------------------
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://dsouzacrystal:dsouzacrystal2003@cluster0.uufjq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
MONGO_DB_NAME = "scrum_db"  # Simpler database name
BOARDS_COLLECTION = "fastapi_boards"
CONVERSATIONS_COLLECTION = "fastapi_conversations"

# Initialize FastAPI app
app = FastAPI()

# Initialize MongoDB client (as a dependency for FastAPI)
def get_mongo_client():
    client = MongoClient(MONGO_URI)
    try:
        yield client
    finally:
        client.close()

# Get the database instance
def get_database(client: MongoClient = Depends(get_mongo_client)):
    return client[MONGO_DB_NAME]

# --------------------------------------------------------------------------------
# 2) MongoDB Helper Functions (Simplified)
# --------------------------------------------------------------------------------
async def store_board_fastapi(db, board_id: str, name: str, board_type: str):
    """Store a simplified board document into MongoDB."""
    board_doc = {
        "board_id": board_id,
        "name": name,
        "type": board_type,
        "created_at": datetime.utcnow()
    }
    try:
        await db[BOARDS_COLLECTION].insert_one(board_doc)
        return {"message": f"Board '{name}' stored successfully."}
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail=f"Board with id '{board_id}' already exists.")

async def get_boards_fastapi(db):
    """Fetch all stored boards."""
    boards = []
    async for board in db[BOARDS_COLLECTION].find():
        board["_id"] = str(board["_id"])  # Convert ObjectId to string for JSON response
        boards.append(board)
    return boards

class Conversation(BaseModel):
    user_id: str
    message: str
    timestamp: datetime = datetime.utcnow()

async def store_conversation_fastapi(db, conversation: Conversation):
    """Store a simplified conversation message."""
    try:
        await db[CONVERSATIONS_COLLECTION].insert_one(conversation.dict())
        return {"message": "Conversation message stored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing conversation: {e}")

async def get_conversations_fastapi(db, user_id: str):
    """Retrieve recent conversations for a user."""
    conversations = []
    async for convo in db[CONVERSATIONS_COLLECTION].find({"user_id": user_id}).sort("timestamp", -1).limit(5):
        convo["_id"] = str(convo["_id"])
        conversations.append(convo)
    return conversations

# --------------------------------------------------------------------------------
# 3) Simplified Bot Logic
# --------------------------------------------------------------------------------
class ScrumBot:
    def __init__(self):
        self.greeting = "Hello! How can I help with the scrum today?"

    async def process_message(self, user_id: str, message: str, db):
        """Process user messages and return a response."""
        if "hello" in message.lower() or "hi" in message.lower():
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User said: {message}"))
            return self.greeting
        elif "standup" in message.lower():
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User asked about standup: {message}"))
            return "Okay, let's start a simplified standup. What did you work on yesterday?"
        elif "yesterday" in message.lower():
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User update for yesterday: {message}"))
            return "Great. What are you planning to work on today?"
        elif "today" in message.lower():
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User plans for today: {message}"))
            return "Are there any blockers or impediments?"
        elif "blocker" in message.lower() or "impediment" in message.lower():
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User mentioned a blocker: {message}"))
            return "Thank you for the update. Let's see how we can resolve that."
        elif "done" in message.lower() or "finished" in message.lower():
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User finished standup: {message}"))
            return "Thanks for the standup update!"
        else:
            await store_conversation_fastapi(db, Conversation(user_id=user_id, message=f"User said: {message}"))
            return "I can help with basic scrum updates."

# Initialize the bot
scrum_bot = ScrumBot()

# --------------------------------------------------------------------------------
# 4) FastAPI Endpoints
# --------------------------------------------------------------------------------
class BoardCreate(BaseModel):
    board_id: str
    name: str
    board_type: str

@app.post("/boards/", response_model=Dict)
async def create_board(board: BoardCreate, db: MongoClient = Depends(get_database)):
    """Endpoint to create and store a new board."""
    return await store_board_fastapi(db, board.board_id, board.name, board.board_type)

@app.get("/boards/", response_model=List[Dict])
async def get_all_boards(db: MongoClient = Depends(get_database)):
    """Endpoint to retrieve all stored boards."""
    return await get_boards_fastapi(db)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: MongoClient = Depends(get_database)):
    """Endpoint to send a message to the scrum bot."""
    response = await scrum_bot.process_message(request.user_id, request.message, db)
    return {"response": response}

@app.get("/conversations/{user_id}", response_model=List[Dict])
async def get_user_conversations(user_id: str, db: MongoClient = Depends(get_database)):
    """Endpoint to retrieve recent conversations for a specific user."""
    return await get_conversations_fastapi(db, user_id)