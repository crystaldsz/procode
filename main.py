import os
import json
import logging
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
# from scrum_agent import AIScrumMaster, get_boards  # Import your existing logic
from botbuilder.core import BotFrameworkAdapterSettings, BotFrameworkAdapter, TurnContext, ActivityHandler
from botbuilder.schema import Activity, ActivityTypes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load Microsoft Teams Bot Credentials
MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD")

# In-memory conversation state storage
conversations = {}

# Define Pydantic models for incoming Teams messages
class TeamsFrom(BaseModel):
    id: str

class TeamsConversation(BaseModel):
    id: str

class TeamsMessage(BaseModel):
    from_: TeamsFrom = Field(..., alias="from")
    conversation: TeamsConversation
    text: str
    serviceUrl: str
    id: str

@app.get("/")
def read_root():
    return {"message": "FastAPI Teams Bot is running!"}

# 🔹 STEP 1: Authenticate with Microsoft Bot Framework API
def get_auth_token():
    """Get an authentication token for Microsoft Teams API"""
    token_url = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": MICROSOFT_APP_ID,
        "client_secret": MICROSOFT_APP_PASSWORD,
        "scope": "https://api.botframework.com/.default"
    }

    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    logger.error(f"Failed to get auth token: {response.text}")
    return None

# 🔹 STEP 2: Send a response to Microsoft Teams
def send_teams_response(activity, response_text):
    """Send response back to Microsoft Teams."""
    auth_token = get_auth_token()
    if not auth_token:
        logger.error("Failed to get auth token, cannot send response.")
        return False

    service_url = activity.get("serviceUrl", "")
    conversation_id = activity.get("conversation", {}).get("id", "")
    message_id = activity.get("id", "")

    response_url = f"{service_url}/v3/conversations/{conversation_id}/activities/{message_id}"
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

    response_payload = {
        "type": "message",
        "text": response_text
    }

    logger.info(f"📤 Sending message to Teams: {response_text}")
    logger.info(f"🔗 POST {response_url}")

    response = requests.post(response_url, json=response_payload, headers=headers)

    if response.status_code == 200:
        logger.info("✅ Successfully sent message to Teams!")
        return True
    else:
        logger.error(f"❌ Failed to send message to Teams: {response.text}")
        return False

# 🔹 STEP 3: Process Messages
def process_message(user_id, text, conversation_id):
    """
    Process the incoming user message and return a response.
    """
    logger.info(f"📝 Processing message from user {user_id}: {text}")

    # Initialize conversation state if not present
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            # "bot": AIScrumMaster(user_id), # Uncomment when you have your class
            "credentials": None,
            "board_id": None,
            "answered_steps": {},
            "member_step": {}
        }

    conv_state = conversations[conversation_id]
    # bot = conv_state["bot"] # Uncomment when you have your class

    # Step 1: Handle basic greetings
    if text.lower() in ["hi", "hello"]:
        return "Hello! 👋 I'm your Scrum Bot. How can I assist you today?"

    # Step 2: Help command
    elif text.lower() == "help":
        return "Available Commands:\n• 'hi' - Greet the bot\n• 'help' - Show available commands\n• 'start' - Begin standup"

    # Step 3: Start standup
    elif text.lower() == "start":
        return "Please provide your Jira credentials in this format:\nJIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN"

    # Step 4: Handle Jira credentials
    elif conv_state["credentials"] is None:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) == 3:
            jira_url, jira_email, jira_api_token = parts
            conv_state["credentials"] = {
                "JIRA_URL": jira_url,
                "JIRA_EMAIL": jira_email,
                "JIRA_API_TOKEN": jira_api_token
            }

            # Retrieve Jira boards
            # boards = get_boards() # Uncomment when you have your function
            boards = [{"id": 1, "name": "Board 1"}, {"id": 2, "name": "Board 2"}] # Placeholder
            if not boards:
                return "No boards found for your Jira account. Please check your credentials."
            else:
                board_options = "\n".join([f"{board['id']}: {board.get('name', 'Unknown')}" for board in boards])
                return f"Here are your available boards:\n{board_options}\nReply with the board ID you'd like to select."

        else:
            return "Invalid format. Please provide Jira credentials as: JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN"

    # Step 5: Handle board selection
    elif conv_state["board_id"] is None and text.isdigit():
        board_id = int(text)
        conv_state["board_id"] = board_id

        # if bot.initialize_sprint_data(board_id): # Uncomment when you have your class
        #     team_members = list(bot.team_members)
        team_members = ["User1", "User2"] # Placeholder
        if team_members:
            first_member = team_members[0]
            conv_state.setdefault("member_step", {})
            if first_member not in conv_state["member_step"]:
                conv_state["member_step"][first_member] = 1
            current_step = conv_state["member_step"][first_member]
            # return bot.generate_question(first_member, current_step) # Uncomment when you have your class
            return f"Starting standup questions for {first_member} (Step {current_step})"
        else:
            return f"Standup started on board {board_id}, but no assigned team members found."

    else:
        # bot.add_user_response(user_id, text) # Uncomment when you have your class
        # member = list(bot.team_members)[0] if bot.team_members else user_id # Uncomment when you have your class
        member = user_id # Placeholder
        conv_state.setdefault("member_step", {})
        conv_state["member_step"][member] = conv_state["member_step"].get(member, 0) + 1
        current_step = conv_state["member_step"][member]
        conv_state.setdefault("answered_steps", {}).setdefault(member, {})
        conv_state["answered_steps"][member][current_step] = True

        # return bot.generate_question(member, len(bot.conversation_history)) # Uncomment when you have your class
        return f"Processed your response. Next question for {member} (Step {current_step})"

# 🔹 STEP 4: Handle Messages from Teams
@app.post("/api/messages")
async def messages(msg: TeamsMessage):
    """Handle messages from Microsoft Teams."""
    logger.info(f"📩 Received Teams message: {msg.text}")

    # Extract details from the message
    user_id = msg.from_.id
    conversation_id = msg.conversation.id
    text = msg.text.strip()

    # Process the message
    response_text = process_message(user_id, text, conversation_id)

    # Send the response back to Teams
    send_teams_response(msg.dict(), response_text)

    return JSONResponse(content={"status": "Message processed successfully"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)