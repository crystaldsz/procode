import os
import logging
import json
from typing import Dict, Any, List, Optional
import asyncio

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field # Keep Pydantic for request validation if needed externally

# --- BotBuilder SDK Imports ---
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
    ActivityHandler,
    MessageFactory,
    ConversationState,
    UserState,
    # MemoryStorage # Replaced with FileStorage for persistent local testing
)
# ADDED: FileStorage for persistent local testing state
from botbuilder.core.storage import FileStorage
from botbuilder.schema import (
    Activity,
    ActivityTypes,
    ChannelAccount,
    ConversationAccount,
    Mention
)
# To get mentions correctly in Teams
from botbuilder.core.teams import TeamsActivityHandler, TeamsInfo


# --- Your Scrum Logic Placeholder ---
# Replace these with your actual Jira/AI logic later
# from scrum_agent import AIScrumMaster, get_boards

# Placeholder function for getting boards
async def get_boards_placeholder(credentials: Dict[str, str]) -> List[Dict[str, Any]]:
    """Placeholder: Simulates fetching Jira boards."""
    # Simulate checking credentials - replace with actual validation if needed
    if not credentials or not credentials.get("JIRA_URL") or not credentials.get("JIRA_EMAIL") or not credentials.get("JIRA_API_TOKEN"):
        logger.warning("get_boards_placeholder called with incomplete credentials.")
        # Optionally raise an error or return empty list based on desired handling
        # raise ValueError("Incomplete Jira credentials provided.")
        return []
    logger.info(f"Simulating fetching boards with creds for email: {credentials.get('JIRA_EMAIL')}")
    await asyncio.sleep(0.5) # Simulate network delay
    # In a real scenario, use credentials to make an API call to Jira
    return [{"id": "10001", "name": "Project Alpha Board"}, {"id": "10002", "name": "Team Phoenix Sprint Board"}]

# Placeholder function for getting team members (replace with actual logic)
async def get_team_members_placeholder(board_id: str, credentials: Dict[str, str]) -> List[str]:
    """Placeholder: Simulates fetching team members for a board."""
    if not credentials: # Add basic check
        logger.warning("get_team_members_placeholder called without credentials.")
        return []
    logger.info(f"Simulating fetching members for board {board_id}")
    await asyncio.sleep(0.2)
    # In a real scenario, query Jira based on board/project settings
    return ["Alice", "Bob", "Charlie"] # Example team members

# Placeholder for generating questions (replace with AI logic)
def generate_question_placeholder(member_name: str, step: int, total_steps: int = 3) -> str:
    """Placeholder: Generates standup questions."""
    questions = [
        f"@{member_name}, what did you accomplish yesterday?",
        f"@{member_name}, what are you planning to work on today?",
        f"@{member_name}, do you have any blockers?"
    ]
    if 1 <= step <= total_steps:
         # Add mention manually for clarity in placeholder
        return f"{questions[step-1]}"
    elif step > total_steps:
        return f"Thanks @{member_name}, that's all for you!"
    return f"Something went wrong asking question {step} for @{member_name}."

# Placeholder for processing response (replace with AI logic)
def process_user_response_placeholder(member_name: str, step: int, response_text: str):
    """Placeholder: Processes the user's standup answer."""
    logger.info(f"Processing response from {member_name} (Step {step}): {response_text}")
    # In a real scenario, you might store this, analyze it, etc.
    pass
# --- End Placeholders ---


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Scrum Bot for Teams", description="Handles standup interactions in Microsoft Teams.")

# --- Bot Framework Setup ---
MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD")

# Log whether credentials are found (useful for debugging auth issues)
if not MICROSOFT_APP_ID or not MICROSOFT_APP_PASSWORD:
    logger.warning("Microsoft App ID or Password not found in environment variables. Adapter will run without authentication checks (OK for Emulator testing without credentials).")
else:
    logger.info("Microsoft App ID and Password loaded from environment variables.")


SETTINGS = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# --- State Management ---
# Use FileStorage for simple local persistence during testing across restarts
# REMEMBER: Add 'bot_state.json' to your .gitignore file!
try:
    STORAGE = FileStorage("bot_state.json") # Creates/uses a file to store state
    logger.info("Using FileStorage for state persistence (bot_state.json)")
except Exception as e:
    # Fallback if FileStorage fails (e.g., permissions issue)
    logger.error(f"Failed to initialize FileStorage, falling back to MemoryStorage: {e}")
    from botbuilder.core import MemoryStorage
    STORAGE = MemoryStorage()
    logger.warning("Using MemoryStorage for state due to FileStorage error.")

# ConversationState stores data related to the conversation (e.g., current standup state)
CONVERSATION_STATE = ConversationState(STORAGE)
# UserState stores data related to the user (e.g., their Jira credentials - use securely!)
# Note: Storing secrets like API tokens directly in bot state is risky. Consider more secure methods.
USER_STATE = UserState(STORAGE)

# Create property accessors for easy state access
conversation_data_accessor = CONVERSATION_STATE.create_property("ConversationData")
user_profile_accessor = USER_STATE.create_property("UserProfile")

# --- Bot Logic Handler ---
class ScrumBot(TeamsActivityHandler): # Inherit from TeamsActivityHandler for Teams-specific features
    def __init__(self, conv_state: ConversationState, user_state: UserState, conv_accessor, user_accessor):
        self.conversation_state = conv_state
        self.user_state = user_state
        self.conversation_data_accessor = conv_accessor
        self.user_profile_accessor = user_accessor
        self.standup_questions_total = 3 # Total standard questions per person

    async def on_turn(self, turn_context: TurnContext):
        # Override on_turn to load/save state automatically
        await super().on_turn(turn_context)
        # Save any state changes after the turn is processed
        # Use run_backgroundtasks=True for potentially better performance but less predictable save timing
        await self.conversation_state.save_changes(turn_context, force=False)
        await self.user_state.save_changes(turn_context, force=False)
        logger.debug("State changes saved.") # Add debug log for state saving

    async def on_message_activity(self, turn_context: TurnContext):
        """Handles incoming message activities."""
        # Get conversation state (initialize if first time)
        conv_data = await self.conversation_data_accessor.get(turn_context, lambda: {"standup_active": False, "current_member_index": -1, "team_members": [], "member_answers": {}, "current_step": {}})
        # Get user profile state (initialize if first time)
        user_profile = await self.user_profile_accessor.get(turn_context, lambda: {"credentials": None, "board_id": None})

        # ADDED: Log state at the beginning of the turn for debugging
        logger.info(f"START Turn - User: {turn_context.activity.from_property.id}, Conv: {turn_context.activity.conversation.id}")
        logger.info(f"START Turn - Conv Data: {json.dumps(conv_data)}") # Dump state for clarity
        logger.info(f"START Turn - User Profile: {json.dumps(user_profile)}") # Dump state for clarity


        # IMPORTANT: Remove bot mention text from the message
        cleaned_text = TurnContext.remove_recipient_mention(turn_context.activity)
        text = cleaned_text.strip().lower() if cleaned_text else ""

        # Check if the message is from a group chat and if the bot was mentioned
        mentioned = False
        if turn_context.activity.conversation.is_group:
             mentions = turn_context.activity.get_mentions()
             bot_mentioned_in_group = any(m.mentioned.id == turn_context.activity.recipient.id for m in mentions)

             if not bot_mentioned_in_group:
                 logger.info("Bot not mentioned in group chat, ignoring message.")
                 return # Don't respond if not mentioned in a group chat
             else:
                 mentioned = True # Explicitly set for clarity if needed later

        logger.info(f"📩 Received message: '{turn_context.activity.text}' (Is Group: {turn_context.activity.conversation.is_group}, Mentioned: {mentioned}, Processed Text: '{text}')")
        # Basic info logged previously, adding more detail here

        response_text = "Sorry, I didn't understand that. Type 'help' for commands." # Default response

        # --- Command Handling ---
        if text == "hi" or text == "hello":
            response_text = "Hello! 👋 I'm your AI Scrum Bot. Ready to help with standups. Type 'help' to see what I can do."

        elif text == "help":
            response_text = (
                "Available Commands:\n"
                "• `hi` or `hello` - Greet the bot\n"
                "• `help` - Show this help message\n"
                "• `set credentials` - Start the process to set your Jira credentials (URL, Email, API Token)\n"
                "• `select board` - Choose the Jira board for the standup (after setting credentials)\n"
                "• `start standup` - Begin the standup questions for the selected board\n"
                "• `status` - Show the current standup status\n"
                "• `reset` - Reset the standup state for this conversation"
            )

        elif text == "set credentials":
             # Initiate credential setting - ask for URL first
            conv_data["next_credential_step"] = "url" # Track expected input
            response_text = "Okay, please provide your Jira URL (e.g., https://your-domain.atlassian.net):"
            logger.info("Command 'set credentials': Setting next_credential_step=url") # ADDED Log

        elif conv_data.get("next_credential_step") == "url":
            logger.info(f"Processing step 'url', received text: {text}") # ADDED Log
            # Basic validation example (can be more robust)
            if not text.startswith("http://") and not text.startswith("https://"):
                response_text = "Invalid URL format. Please include http:// or https://."
                # Keep expecting url
                logger.warning("Invalid URL format provided for Jira URL.")
            else:
                user_profile["credentials"] = {"JIRA_URL": text}
                conv_data["next_credential_step"] = "email"
                response_text = "Got the URL. Now, please provide your Jira Email:"
                logger.info("Step 'url' processed. Setting next_credential_step=email") # ADDED Log

        elif conv_data.get("next_credential_step") == "email":
            logger.info(f"Processing step 'email', received text: {text}") # ADDED Log
            if user_profile.get("credentials"):
                 # Basic email validation example
                if "@" not in text or "." not in text:
                     response_text = "Invalid email format. Please provide a valid email address."
                     # Keep expecting email
                     logger.warning("Invalid email format provided for Jira Email.")
                else:
                    user_profile["credentials"]["JIRA_EMAIL"] = text
                    conv_data["next_credential_step"] = "token"
                    response_text = "Got the Email. Now, please provide your Jira API Token:"
                    logger.info("Step 'email' processed. Setting next_credential_step=token") # ADDED Log
            else:
                response_text = "Please start with `set credentials` first."
                logger.warning("Credential step 'email' but no credentials found in profile. Resetting step.") # ADDED Log
                conv_data.pop("next_credential_step", None) # Reset step tracking

        elif conv_data.get("next_credential_step") == "token":
            logger.info("Processing step 'token', received text: [REDACTED]") # ADDED Log - Avoid logging tokens
            if user_profile.get("credentials"):
                 # Basic token validation (check if not empty)
                if not text:
                    response_text = "API Token cannot be empty. Please provide your Jira API Token:"
                    # Keep expecting token
                    logger.warning("Empty API token provided.")
                else:
                    user_profile["credentials"]["JIRA_API_TOKEN"] = text # Store securely in production!
                    conv_data.pop("next_credential_step", None) # Clear tracking
                    logger.info(f"Credentials set for user {turn_context.activity.from_property.id}. Cleared next_credential_step.") # ADDED Log
                    response_text = "✅ Credentials saved! You can now use `select board`."
            else:
                response_text = "Please start with `set credentials` first."
                logger.warning("Credential step 'token' but no credentials found in profile. Resetting step.") # ADDED Log
                conv_data.pop("next_credential_step", None)

        elif text == "select board":
            if not user_profile.get("credentials"):
                response_text = "Please set your Jira credentials first using `set credentials`."
            else:
                logger.info("Command 'select board': Attempting to fetch boards.") # ADDED Log
                try:
                    boards = await get_boards_placeholder(user_profile["credentials"])
                    if not boards:
                        response_text = "Could not find any Jira boards associated with your account. Please check your credentials and permissions."
                        logger.warning("No boards found for the user's credentials.")
                    else:
                        board_options = "\n".join([f"• `{board['id']}`: {board.get('name', 'Unknown Name')}" for board in boards])
                        response_text = f"Found these boards:\n{board_options}\n\nPlease reply with the board ID you want to use (e.g., `use board 10001`)."
                        conv_data["expecting_board_id"] = True
                        logger.info(f"Boards fetched successfully. Setting expecting_board_id=True") # ADDED Log
                except Exception as e:
                    logger.error(f"Error fetching boards: {e}", exc_info=True) # Log traceback
                    response_text = "Sorry, I encountered an error trying to fetch your Jira boards."

        elif text.startswith("use board ") and conv_data.get("expecting_board_id"):
            logger.info(f"Processing 'use board', received text: {text}") # ADDED Log
            try:
                board_id = text.split("use board ")[1].strip()
                if not board_id: # Check if ID is empty after split
                     raise IndexError("Board ID cannot be empty.")
                # You might want to add validation here to check if it's one of the listed IDs from get_boards
                user_profile["board_id"] = board_id
                conv_data.pop("expecting_board_id", None)
                response_text = f"✅ Board `{board_id}` selected. You can now `start standup`."
                logger.info(f"User {turn_context.activity.from_property.id} selected board {board_id}. Cleared expecting_board_id.") # ADDED Log
            except IndexError:
                 response_text = "Invalid format or empty ID. Please use `use board <board_id>`."
                 logger.warning(f"Invalid 'use board' format or empty ID: {text}") # ADDED Log

        elif text == "start standup":
            if not user_profile.get("board_id"):
                response_text = "Please select a board first using `select board`."
            elif not user_profile.get("credentials"): # Add check for credentials too
                 response_text = "Please set your Jira credentials first using `set credentials`."
            elif conv_data.get("standup_active"):
                response_text = "A standup is already in progress. Use `status` to see who's next."
            else:
                logger.info("Command 'start standup': Attempting to start.") # ADDED Log
                try:
                    team_members = await get_team_members_placeholder(user_profile["board_id"], user_profile["credentials"])
                    if not team_members:
                         response_text = f"No team members found for board `{user_profile['board_id']}`. Cannot start standup."
                         logger.warning(f"No team members found for board {user_profile['board_id']}.")
                    else:
                        conv_data["standup_active"] = True
                        conv_data["team_members"] = team_members
                        conv_data["current_member_index"] = 0
                        conv_data["member_answers"] = {member: {} for member in team_members} # Init answers dict
                        conv_data["current_step"] = {member: 1 for member in team_members} # Init steps dict

                        first_member = conv_data["team_members"][0]
                        current_step_for_member = conv_data["current_step"][first_member]

                        response_text = f"🚀 Standup started for board `{user_profile['board_id']}` with members: {', '.join(team_members)}.\n\nLet's begin with {first_member}!"
                        question = generate_question_placeholder(first_member, current_step_for_member, self.standup_questions_total)

                        # Send the start message first, then the question
                        await turn_context.send_activity(MessageFactory.text(response_text))
                        response_text = question # The final response for this turn will be the first question

                        logger.info(f"Standup started. Asking {first_member}, step {current_step_for_member}.") # Refined Log

                except Exception as e:
                    logger.error(f"Error starting standup: {e}", exc_info=True) # Log traceback
                    response_text = "Sorry, I encountered an error trying to start the standup."
                    # Reset potentially inconsistent state
                    conv_data["standup_active"] = False
                    conv_data["current_member_index"] = -1


        elif conv_data.get("standup_active"):
            # --- Handle Standup Responses ---
            current_index = conv_data.get("current_member_index", -1)
            if 0 <= current_index < len(conv_data["team_members"]):
                current_member = conv_data["team_members"][current_index]
                current_step = conv_data["current_step"].get(current_member, 1)

                # Simplistic check: Assuming the incoming message is the answer from the current member
                # TODO: Improve this in production (e.g., check turn_context.activity.from_property.id against expected member ID if possible)
                logger.info(f"Standup active: Processing text as answer from {current_member} for step {current_step}")

                # Process the user's response (placeholder)
                process_user_response_placeholder(current_member, current_step, text)
                # Ensure nested dict exists before assignment
                if current_member not in conv_data["member_answers"]:
                    conv_data["member_answers"][current_member] = {}
                conv_data["member_answers"][current_member][current_step] = text # Store the answer

                # Move to the next step for the current member
                next_step = current_step + 1
                conv_data["current_step"][current_member] = next_step

                if next_step <= self.standup_questions_total:
                    # Ask next question for the same member
                    response_text = generate_question_placeholder(current_member, next_step, self.standup_questions_total)
                    logger.info(f"Asking {current_member} next question, step {next_step}")
                else:
                    # Finished with this member, move to the next
                    logger.info(f"Finished questions for {current_member}")
                    next_index = current_index + 1
                    conv_data["current_member_index"] = next_index

                    if next_index < len(conv_data["team_members"]):
                        # Start questions for the next member
                        next_member = conv_data["team_members"][next_index]
                        next_member_step = conv_data["current_step"].get(next_member, 1) # Should be 1
                        response_text = generate_question_placeholder(next_member, next_member_step, self.standup_questions_total)
                        logger.info(f"Moving to next member: {next_member}, step {next_member_step}")
                    else:
                        # Standup finished!
                        response_text = "🎉 Standup complete for all members! Great job team!"
                        logger.info("Standup finished.")
                        # Optionally summarize answers here
                        # Reset standup state
                        conv_data["standup_active"] = False
                        conv_data["current_member_index"] = -1
                        # Keep answers/steps for this session, or clear them:
                        # conv_data["member_answers"] = {}
                        # conv_data["current_step"] = {}

            else:
                # Should not happen if standup_active is true, but handle defensively
                response_text = "Standup is marked active, but I'm unsure whose turn it is. Use `reset` or `status`."
                logger.warning("Standup active but index is invalid.")

        elif text == "status":
            if conv_data.get("standup_active"):
                current_index = conv_data.get("current_member_index", -1)
                if 0 <= current_index < len(conv_data["team_members"]):
                     current_member = conv_data["team_members"][current_index]
                     current_step = conv_data["current_step"].get(current_member, 1)
                     response_text = f"Standup is active on board `{user_profile.get('board_id', 'N/A')}`. Currently waiting for @{current_member} to answer question #{current_step}."
                else:
                     response_text = "Standup is active, but the current member index seems off."
            else:
                response_text = f"No standup currently active. Board selected: `{user_profile.get('board_id', 'None')}`. Use `start standup` to begin."
            logger.info(f"Command 'status': Responded with current status.") # ADDED Log

        elif text == "reset":
            # Reset conversation state related to standup
            conv_data["standup_active"] = False
            conv_data["current_member_index"] = -1
            conv_data["team_members"] = []
            conv_data["member_answers"] = {}
            conv_data["current_step"] = {}
            conv_data.pop("next_credential_step", None) # Clear credential step tracking
            conv_data.pop("expecting_board_id", None) # Clear board selection tracking

            # Reset user state too (credentials, board_id) for this user in this turn
            user_profile["credentials"] = None
            user_profile["board_id"] = None

            response_text = "🔄 Standup state, credentials, and board selection have been reset for this conversation."
            logger.info("Command 'reset': Conversation and relevant user state reset.") # Refined Log


        # ADDED: Log state *before* sending response (and before state save in on_turn)
        logger.info(f"END Turn - Determined Response: '{response_text}'")
        logger.info(f"END Turn - Conv Data: {json.dumps(conv_data)}") # Dump state for clarity
        logger.info(f"END Turn - User Profile: {json.dumps(user_profile)}") # Dump state for clarity

        # --- Send the response ---
        # Check if we already sent a response (e.g., asking the first question immediately after 'start standup')
        if not turn_context.responded:
             await turn_context.send_activity(MessageFactory.text(response_text))


    async def on_members_added_activity(self, members_added: List[ChannelAccount], turn_context: TurnContext):
        """Greet new members when they are added to the conversation."""
        for member in members_added:
            # Greet anyone added that isn't the bot itself
            if member.id != turn_context.activity.recipient.id:
                logger.info(f"Member added: {member.name} ({member.id})")
                await turn_context.send_activity(
                    MessageFactory.text(f"Welcome, {member.name}! I'm the Scrum Bot. Type 'help' to see what I can do.")
                )

# --- Create the Bot Instance ---
BOT = ScrumBot(CONVERSATION_STATE, USER_STATE, conversation_data_accessor, user_profile_accessor)

# --- FastAPI Endpoint ---
@app.post("/api/messages")
async def handle_messages(request: Request):
    """Main endpoint to receive messages from the Bot Framework Service."""
    if "application/json" not in request.headers.get("Content-Type", ""):
        logger.error("Request content type is not application/json")
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

    body = await request.json()
    # logger.debug(f"Received raw body: {json.dumps(body, indent=2)}") # Keep commented unless needed

    # Create an Activity object from the incoming request body
    activity = Activity().deserialize(body)
    # logger.debug(f"Deserialized activity type: {activity.type}") # Keep commented unless needed

    # Get the Authorization header from the request
    auth_header = request.headers.get("Authorization", "")

    try:
        # Process the activity using the Bot Framework Adapter.
        # This handles authentication and directs the activity to the BOT's handler
        # BOT.on_turn handles calling the right method (on_message_activity, etc.) AND saving state
        response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)

        # The adapter's response might contain information for the Bot Framework Service
        if response:
            logger.info(f"Adapter process_activity completed with status: {response.status}")
            return Response(content=response.body, status_code=response.status, media_type="application/json")
        else:
            # If the adapter doesn't return a specific response (common for message activities), return HTTP 202 Accepted
            logger.info("Adapter process_activity completed successfully (no specific response body needed). Returning 202 Accepted.")
            return Response(status_code=202) # Use 202 Accepted for async message handling

    except Exception as e:
        logger.exception(f"Critical error processing activity: {e}") # Log the full traceback
        # Return an HTTP 500 Internal Server Error if something goes wrong
        raise HTTPException(status_code=500, detail="Internal Server Error processing activity") # Avoid sending raw error details

@app.get("/")
def read_root():
    return {"message": "AI Scrum Bot for Teams using FastAPI and BotBuilder SDK is running!"}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower() # Allow setting log level via env var
    logger.info(f"Starting FastAPI server on host 0.0.0.0 port {port} with log level {log_level}")

    # Use reload=True for development only. Ensure file is named 'main.py' or adjust "main:app".
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level=log_level, reload=True) # Set reload=False for production