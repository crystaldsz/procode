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
    MemoryStorage # Use MemoryStorage for simple testing, replace for production
)
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
    logger.info(f"Simulating fetching boards with creds: {credentials.get('JIRA_EMAIL')}")
    await asyncio.sleep(0.5) # Simulate network delay
    # In a real scenario, use credentials to make an API call to Jira
    return [{"id": "10001", "name": "Project Alpha Board"}, {"id": "10002", "name": "Team Phoenix Sprint Board"}]

# Placeholder function for getting team members (replace with actual logic)
async def get_team_members_placeholder(board_id: str, credentials: Dict[str, str]) -> List[str]:
    """Placeholder: Simulates fetching team members for a board."""
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

if not MICROSOFT_APP_ID or not MICROSOFT_APP_PASSWORD:
    logger.error("FATAL ERROR: Microsoft App ID or Password not found in environment variables.")
    # You might want to exit or raise an exception here in a real deployment
    # For now, we'll let it potentially fail later during adapter use.

SETTINGS = BotFrameworkAdapterSettings(MICROSOFT_APP_ID, MICROSOFT_APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)

# --- State Management ---
# Use MemoryStorage for development/testing. Replace with CosmosDbPartitionedStorage, BlobStorage, etc., for production.
MEMORY_STORAGE = MemoryStorage()
# ConversationState stores data related to the conversation (e.g., current standup state)
CONVERSATION_STATE = ConversationState(MEMORY_STORAGE)
# UserState stores data related to the user (e.g., their Jira credentials - use securely!)
# Note: Storing secrets like API tokens directly in bot state is risky. Consider more secure methods like Azure Key Vault or user-specific secure storage.
USER_STATE = UserState(MEMORY_STORAGE)

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
        await self.conversation_state.save_changes(turn_context, False)
        await self.user_state.save_changes(turn_context, False)

    async def on_message_activity(self, turn_context: TurnContext):
        """Handles incoming message activities."""
        # Get conversation state (initialize if first time)
        # Default dict helps avoid KeyError if properties don't exist yet
        conv_data = await self.conversation_data_accessor.get(turn_context, lambda: {"standup_active": False, "current_member_index": -1, "team_members": [], "member_answers": {}, "current_step": {}})
        # Get user profile state (initialize if first time)
        user_profile = await self.user_profile_accessor.get(turn_context, lambda: {"credentials": None, "board_id": None})

        # IMPORTANT: Remove bot mention text from the message
        # This prevents the bot's name from being processed as part of the command/response
        cleaned_text = TurnContext.remove_recipient_mention(turn_context.activity)
        text = cleaned_text.strip().lower() if cleaned_text else ""

        # Check if the message is from a group chat and if the bot was mentioned
        # In 1:1 chats, mentions are not usually needed/present.
        # In group chats, usually respond only when mentioned.
        mentioned = False
        if turn_context.activity.conversation.is_group:
             mentions = turn_context.activity.get_mentions()
             for mention in mentions:
                 # Check if the mention's ID matches the bot's ID
                 if mention.mentioned.id == turn_context.activity.recipient.id:
                     mentioned = True
                     break
             if not mentioned:
                 logger.info("Bot not mentioned in group chat, ignoring message.")
                 return # Don't respond if not mentioned in a group chat

        logger.info(f"📩 Received message: '{turn_context.activity.text}' (Processed as: '{text}')")
        logger.info(f"Conversation ID: {turn_context.activity.conversation.id}")
        logger.info(f"User ID: {turn_context.activity.from_property.id}")
        logger.info(f"Current conv_data: {conv_data}")
        logger.info(f"Current user_profile: {user_profile}")

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

        elif conv_data.get("next_credential_step") == "url":
            user_profile["credentials"] = {"JIRA_URL": text}
            conv_data["next_credential_step"] = "email"
            response_text = "Got the URL. Now, please provide your Jira Email:"

        elif conv_data.get("next_credential_step") == "email":
            if user_profile.get("credentials"):
                user_profile["credentials"]["JIRA_EMAIL"] = text
                conv_data["next_credential_step"] = "token"
                response_text = "Got the Email. Now, please provide your Jira API Token:"
            else:
                response_text = "Please start with `set credentials` first."
                conv_data.pop("next_credential_step", None) # Reset step tracking

        elif conv_data.get("next_credential_step") == "token":
            if user_profile.get("credentials"):
                user_profile["credentials"]["JIRA_API_TOKEN"] = text # Store securely in production!
                conv_data.pop("next_credential_step", None) # Clear tracking
                logger.info(f"Credentials set for user {turn_context.activity.from_property.id}")
                response_text = "✅ Credentials saved! You can now use `select board`."
            else:
                response_text = "Please start with `set credentials` first."
                conv_data.pop("next_credential_step", None)

        elif text == "select board":
            if not user_profile.get("credentials"):
                response_text = "Please set your Jira credentials first using `set credentials`."
            else:
                try:
                    boards = await get_boards_placeholder(user_profile["credentials"])
                    if not boards:
                        response_text = "Could not find any Jira boards associated with your account. Please check your credentials and permissions."
                    else:
                        board_options = "\n".join([f"• `{board['id']}`: {board.get('name', 'Unknown Name')}" for board in boards])
                        response_text = f"Found these boards:\n{board_options}\n\nPlease reply with the board ID you want to use (e.g., `use board 10001`)."
                        conv_data["expecting_board_id"] = True
                except Exception as e:
                    logger.error(f"Error fetching boards: {e}")
                    response_text = "Sorry, I encountered an error trying to fetch your Jira boards."

        elif text.startswith("use board ") and conv_data.get("expecting_board_id"):
            try:
                board_id = text.split("use board ")[1].strip()
                # You might want to add validation here to check if it's one of the listed IDs
                user_profile["board_id"] = board_id
                conv_data.pop("expecting_board_id", None)
                response_text = f"✅ Board `{board_id}` selected. You can now `start standup`."
                logger.info(f"User {turn_context.activity.from_property.id} selected board {board_id}")
            except IndexError:
                 response_text = "Invalid format. Please use `use board <board_id>`."

        elif text == "start standup":
            if not user_profile.get("board_id"):
                response_text = "Please select a board first using `select board`."
            elif conv_data.get("standup_active"):
                response_text = "A standup is already in progress. Use `status` to see who's next."
            else:
                try:
                    # Fetch team members for the selected board
                    # Pass credentials securely if needed by the real function
                    team_members = await get_team_members_placeholder(user_profile["board_id"], user_profile["credentials"])
                    if not team_members:
                         response_text = f"No team members found for board `{user_profile['board_id']}`. Cannot start standup."
                    else:
                        conv_data["standup_active"] = True
                        conv_data["team_members"] = team_members
                        conv_data["current_member_index"] = 0
                        conv_data["member_answers"] = {member: {} for member in team_members} # Init answers dict
                        conv_data["current_step"] = {member: 1 for member in team_members} # Init steps dict
                        
                        first_member = conv_data["team_members"][0]
                        current_step_for_member = conv_data["current_step"][first_member]
                        
                        response_text = f"🚀 Standup started for board `{user_profile['board_id']}` with members: {', '.join(team_members)}.\n\nLet's begin with {first_member}!"
                        # Ask the first question immediately after starting
                        question = generate_question_placeholder(first_member, current_step_for_member, self.standup_questions_total)

                        # Send the start message first, then the question
                        await turn_context.send_activity(MessageFactory.text(response_text))
                        # Send the question (this will be the actual response for this turn)
                        response_text = question

                        logger.info(f"Standup started. Asking {first_member}, step {current_step_for_member}.")

                except Exception as e:
                    logger.error(f"Error starting standup: {e}")
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

                # Heuristic: Assume any message during an active standup *might* be an answer
                # More robust: Check if the sender is the person being asked. For now, let's assume it is.
                # In a group chat, this might be noisy. Ideally, only the mentioned user's reply is processed.
                # We'll keep it simple: process the text as the answer for the current member/step.

                # Process the user's response (placeholder)
                process_user_response_placeholder(current_member, current_step, text)
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
                        # Keep answers if needed, or clear them:
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

        elif text == "reset":
            # Reset conversation state related to standup
            conv_data["standup_active"] = False
            conv_data["current_member_index"] = -1
            conv_data["team_members"] = []
            conv_data["member_answers"] = {}
            conv_data["current_step"] = {}
            conv_data.pop("next_credential_step", None)
            conv_data.pop("expecting_board_id", None)
            # Optionally reset user state too (credentials, board_id)
            # await self.user_profile_accessor.delete(turn_context) # This deletes the whole profile
            user_profile["credentials"] = None # Or reset specific parts
            user_profile["board_id"] = None
            response_text = "🔄 Standup state and board selection have been reset for this conversation."
            logger.info("Conversation state reset.")


        # --- Send the response ---
        # Check if we already sent a response (e.g., asking the first question immediately after 'start standup')
        if not turn_context.responded:
             # Prepare mention if needed within the response text (placeholders do this)
             # For group chats, mentioning the user you're addressing is good practice.
             # The generate_question_placeholder includes the @mention.
             # Example of manually adding mention if needed:
             # if conv_data.get("standup_active") and ...:
             #     member_to_mention = ...
             #     member_account = ChannelAccount(id=...) # Need the Teams User ID
             #     mention_entity = Mention(mentioned=member_account, text=f"@{member_to_mention}")
             #     reply_activity = MessageFactory.text(response_text)
             #     reply_activity.entities = [mention_entity]
             #     await turn_context.send_activity(reply_activity)
             # else:
             #     await turn_context.send_activity(MessageFactory.text(response_text))

             # Simple text reply using the determined response_text
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
    # logger.debug(f"Received raw body: {json.dumps(body, indent=2)}") # Careful logging raw body

    # Create an Activity object from the incoming request body
    activity = Activity().deserialize(body)
    # logger.debug(f"Deserialized activity type: {activity.type}")

    # Get the Authorization header from the request
    auth_header = request.headers.get("Authorization", "")

    try:
        # Process the activity using the Bot Framework Adapter.
        # This handles authentication and directs the activity to the BOT's handler (e.g., on_message_activity)
        response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)

        # The adapter's response might contain information for the Bot Framework Service (e.g., invoke responses)
        if response:
            # Log the response status code if available
            logger.info(f"Adapter process_activity completed with status: {response.status}")
            # Return the adapter's response content and status code
            # Important for invoke activities or specific Bot Framework protocols
            return Response(content=response.body, status_code=response.status, media_type="application/json")
        else:
            # If the adapter doesn't return a specific response (common for message activities), return HTTP 202 Accepted
            # This acknowledges receipt of the message activity. The actual reply is sent asynchronously via turn_context.send_activity.
            logger.info("Adapter process_activity completed successfully (no specific response needed). Returning 202 Accepted.")
            return Response(status_code=202) # Use 202 Accepted for async message handling

    except Exception as e:
        logger.exception(f"Error processing activity: {e}") # Log the full traceback
        # Return an HTTP 500 Internal Server Error if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "AI Scrum Bot for Teams using FastAPI and BotBuilder SDK is running!"}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting FastAPI server on host 0.0.0.0 port {port}")
    # Use reload=True for development only
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info") # Make sure your file is named main.py or adjust here