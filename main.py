import os
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import google.generativeai as genai
import requests
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Bot Framework imports
from botbuilder.core import BotFrameworkAdapterSettings, TurnContext, BotFrameworkAdapter
from botbuilder.schema import Activity, ActivityTypes, ConversationReference

# --------------------------------------------------------------------------------
# 1) Load environment variables and configure APIs
# --------------------------------------------------------------------------------
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://dsouzacrystal:dsouzacrystal2003@cluster0.uufjq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

# JIRA Configuration
JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
jira_auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
jira_headers = {"Accept": "application/json"}

# Gemini Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Bot Framework Configuration
BOT_APP_ID = os.getenv("MicrosoftAppId", "3a62a6ac-053d-4cb7-8781-4786a695705b")  # Replace with your actual App ID
BOT_APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")  # Replace with your actual App Password

# Configure Bot Framework adapter
BOT_SETTINGS = BotFrameworkAdapterSettings(BOT_APP_ID, BOT_APP_PASSWORD)
BOT_ADAPTER = BotFrameworkAdapter(BOT_SETTINGS)

# Store conversation references
CONVERSATION_REFERENCES = {}

# --------------------------------------------------------------------------------
# 2) MongoDB Setup
# --------------------------------------------------------------------------------
client = MongoClient(MONGO_URI)
db = client["jira_db"]
boards_collection = db["boards"]
sprints_collection = db["sprints"]
issues_collection = db["issues"]
users_collection = db["users"]
conversations_collection = db["conversations"]
teams_users_collection = db["teams_users"]  # New collection for Teams users

# --------------------------------------------------------------------------------
# 2.1) MongoDB Helper Functions
# --------------------------------------------------------------------------------
def store_board(board: Dict):
    """Store a Jira board document into MongoDB."""
    board_doc = {
        "board_id": board.get('id'),
        "name": board.get('name'),
        "type": board.get('type'),
        "created_at": datetime.utcnow()
    }
    try:
        boards_collection.insert_one(board_doc)
    except DuplicateKeyError:
        print(f"Board with id {board.get('id')} already exists.")

def store_sprint(sprint: Dict, board_id: int):
    """Store a sprint document into MongoDB."""
    sprint_doc = {
        "sprint_id": sprint.get('id'),
        "board_id": board_id,
        "name": sprint.get('name'),
        "state": sprint.get('state'),
        "start_date": sprint.get('startDate'),
        "end_date": sprint.get('endDate'),
        "goal": sprint.get('goal', 'No goal set'),
        "issues": [issue.get('Key') for issue in sprint.get('issues', [])]
    }
    try:
        sprints_collection.insert_one(sprint_doc)
    except DuplicateKeyError:
        print(f"Sprint with id {sprint.get('id')} already exists.")

def store_issue(issue: Dict, board_id: int, sprint_id: int):
    """Store an issue document into MongoDB."""
    issue_doc = {
        "issue_id": issue.get('Key'),
        "board_id": board_id,
        "sprint_id": sprint_id,
        "summary": issue.get('Summary'),
        "status": issue.get('Status'),
        "assignee": issue.get('Assignee'),
        "story_points": issue.get('story_points', None),
        "created_at": issue.get('Created'),
        "updated_at": issue.get('Updated')
    }
    try:
        issues_collection.insert_one(issue_doc)
    except DuplicateKeyError:
        print(f"Issue with id {issue.get('Key')} already exists.")

def store_user(user_id: str, display_name: str):
    """Store a user document into MongoDB."""
    user_doc = {
        "user_id": user_id,
        "display_name": display_name,
        "created_at": datetime.utcnow()
    }
    try:
        users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        print(f"User with id {user_id} already exists.")

def store_teams_user(teams_user_id: str, name: str, email: str = None):
    """Store a Microsoft Teams user in MongoDB."""
    user_doc = {
        "teams_user_id": teams_user_id,
        "name": name,
        "email": email,
        "created_at": datetime.utcnow()
    }
    try:
        teams_users_collection.update_one(
            {"teams_user_id": teams_user_id},
            {"$set": user_doc},
            upsert=True
        )
    except Exception as e:
        print(f"Error storing Teams user: {e}")

def store_conversation(conversation_doc: dict):
    """Store a conversation document into MongoDB."""
    conversation_doc["date"] = datetime.utcnow()
    conversations_collection.insert_one(conversation_doc)

def get_previous_standups(user_id: str, limit=5):
    """Retrieve recent standup documents from MongoDB for a specific user."""
    cursor = conversations_collection.find({"user_id": user_id}).sort("date", -1).limit(limit)
    return list(cursor)

# --------------------------------------------------------------------------------
# 3) JIRA Integration Functions
# --------------------------------------------------------------------------------
def extract_content_from_adf(content):
    """Extract plain text from Atlassian Document Format (ADF)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if 'text' in content:
            return content['text']
        if 'content' in content:
            return ' '.join(extract_content_from_adf(c) for c in content['content'])
    if isinstance(content, list):
        return ' '.join(extract_content_from_adf(c) for c in content)
    return ''

def get_field_value(issue: Dict, field_name: str) -> str:
    """Extract specific field values with proper fallback."""
    fields = issue.get('fields', {})
    if field_name == 'description':
        content = fields.get('description')
        return extract_content_from_adf(content) if content else "No description available"
    if field_name == 'assignee':
        assignee = fields.get('assignee')
        return assignee.get('displayName') if assignee else "Unassigned"
    if field_name == 'status':
        status = fields.get('status')
        return status.get('name') if status else "Unknown"
    return str(fields.get(field_name, "Not available"))

def get_issue_details(issue: Dict) -> Dict:
    """Return a dictionary with key details about an issue."""
    fields = issue.get('fields', {})
    return {
        'Key': issue.get('key'),
        'Summary': get_field_value(issue, 'summary'),
        'Status': get_field_value(issue, 'status'),
        'Assignee': get_field_value(issue, 'assignee'),
        'Reporter': get_field_value(issue, 'reporter'),
        'Priority': fields.get('priority', {}).get('name', 'Not set'),
        'Issue Type': fields.get('issuetype', {}).get('name', 'Unknown'),
        'Created': fields.get('created', 'Unknown'),
        'Updated': fields.get('updated', 'Unknown'),
        'Description': get_field_value(issue, 'description')
    }

def get_boards() -> List[Dict]:
    """Fetch all available Scrum boards from JIRA."""
    url = f"{JIRA_URL}/rest/agile/1.0/board"
    response = requests.get(url, headers=jira_headers, auth=jira_auth)
    if response.status_code == 200:
        boards = response.json().get('values', [])
        for board in boards:
            store_board(board)
        return boards
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Error fetching boards: {response.text}")

def fetch_sprint_details(board_id: int, include_closed: bool = False) -> List[Dict]:
    """Fetch sprints and their issues for the given board."""
    url = f"{JIRA_URL}/rest/agile/1.0/board/{board_id}/sprint"
    response = requests.get(url, headers=jira_headers, auth=jira_auth)
    sprints_list = []
    if response.status_code == 200:
        for sprint in response.json().get('values', []):
            sprint_id = sprint['id']
            issues_url = f"{JIRA_URL}/rest/agile/1.0/sprint/{sprint_id}/issue"
            issues_response = requests.get(issues_url, headers=jira_headers, auth=jira_auth)
            issues = []
            if issues_response.status_code == 200:
                issues = [get_issue_details(issue) for issue in issues_response.json().get('issues', [])]
            sprint_data = {
                'id': sprint_id,
                'name': sprint.get('name', 'N/A'),
                'state': sprint.get('state', 'N/A'),
                'start_date': sprint.get('startDate', 'N/A'),
                'end_date': sprint.get('endDate', 'N/A'),
                'goal': sprint.get('goal', 'No goal set'),
                'issues': issues
            }
            store_sprint(sprint_data, board_id)
            for issue in issues:
                store_issue(issue, board_id, sprint_id)
            sprints_list.append(sprint_data)
        return sprints_list
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Error fetching sprints: {response.text}")

# --------------------------------------------------------------------------------
# 4) AI Scrum Master Class
# --------------------------------------------------------------------------------
class AIScrumMaster:
    def __init__(self, user_id: str):
        self.user_id = user_id  # To track user-specific data
        self.conversation_history = []
        self.current_sprint = None
        self.team_members = set()
        self.blockers = []
        self.action_items = []
        self.context_cache = {}  # For caching contextual history
        self.current_member_index = 0
        self.conversation_step = 1
        self.standup_started = False
        self.messages = []
        self.nothing_count = 0
        self.conversation_reference = None  # For Teams integration

        # Initialize with a system prompt
        self.system_prompt = (
            "You are an AI Scrum Master named AgileBot. You greet team members warmly, "
            "ask about their tasks, blockers, and updates in a friendly, empathetic, "
            "and concise way. Always maintain a helpful and professional tone and use bullet points when helpful."
        )
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt,
            "timestamp": datetime.utcnow()
        })

        # Load previous standups
        previous_standups = get_previous_standups(self.user_id, limit=3)
        for doc in reversed(previous_standups):
            self.conversation_history.extend(doc.get("messages", []))

    def initialize_sprint_data(self, board_id: int):
        """Initialize sprint data from JIRA."""
        sprints = fetch_sprint_details(board_id, include_closed=False)
        if sprints:
            active_sprints = [s for s in sprints if s['state'] == 'active']
            if active_sprints:
                self.current_sprint = active_sprints[0]
                for issue in self.current_sprint['issues']:
                    assignee = issue.get('Assignee')
                    if assignee and assignee != "Unassigned":
                        self.team_members.add(assignee)
                        store_user(assignee, assignee)
                self.team_members_list = list(self.team_members)
                self.current_member_index = 0
                self.conversation_step = 1
                self.messages = []
                self.standup_started = True
                self.nothing_count = 0
                return True
        return False

    def get_member_tasks(self, member_name: str) -> List[Dict]:
        """Get active tasks for a team member from the current sprint."""
        if not self.current_sprint:
            return []
        return [
            issue for issue in self.current_sprint['issues']
            if issue.get('Assignee') == member_name
        ]

    def build_tasks_context(self, member_name: str) -> str:
        """Build context string for member's tasks."""
        tasks = self.get_member_tasks(member_name)
        if not tasks:
            return "No tasks assigned currently."
        return "\n".join([
            f"- {task['Key']}: {task['Summary']} (Status: {task['Status']})"
            for task in tasks
        ])

    def get_mongo_context(self, member_name: str) -> str:
        # Retrieve the last 5 standup documents for this user
        docs = get_previous_standups(self.user_id, limit=5)
        context_lines = []
        for doc in docs:
            # You can adjust this filtering based on how you store messages.
            for msg in doc.get("messages", []):
                # Optionally, filter messages related to the current member.
                if member_name in msg.get("content", "") or msg.get("role") == "assistant":
                    context_lines.append(msg["content"])
        if context_lines:
            return "\nRelevant Historical Updates:\n" + "\n".join(f"- {line}" for line in context_lines)
        return "No historical updates available."

    def get_contextual_history(self, member_name: str) -> str:
        """Get relevant historical context for the team member."""
        return self.get_mongo_context(member_name)

    def generate_question(self, member_name: str, step: int) -> str:
        """
        Return a fixed Scrum question that's further refined using
        MongoDB context + tasks context + LLM.
        """
        # Standard, fixed Scrum questions
        base_questions = {
            1: f"Hey {member_name}, how are you doing today? How are you feeling about your tasks?",
            2: f"Could you update me on what you accomplished recently, and if you ran into any challenges?",
            3: f"Great, thanks for the update! What's on your agenda for today?",
            4: f"Are there any blockers or issues that you need help with?",
            5: f"Anything else you'd like to add before we wrap up?"
        }
        base_question = base_questions.get(step, "Is there anything else you'd like to discuss?")

        # Build the user's task context (from the current sprint)
        tasks_context = self.build_tasks_context(member_name)
        mongo_context = self.get_mongo_context(member_name)

        # Use the LLM to merge the base question with the retrieved context
        # so the final question feels more personalized and intelligent.
        prompt = f"""
        You are an AI Scrum Master named AgileBot conducting a standup with {member_name} at step {step}.

        Here is the standard Scrum question you should ask:
        "{base_question}"

        Tasks context for {member_name}:
        {tasks_context}

        Historical context from MongoDB:
        {mongo_context}

        Using the above information, generate a single, friendly, and concise question that incorporates all relevant details.
        """

        # Call the Gemini model to generate a refined question
        try:
            refined_question = model.generate_content(prompt).text.strip()
        except Exception as e:
            print(f"Error generating question: {e}")
            refined_question = base_question

        # Fallback if LLM returns something empty (rare edge case)
        if not refined_question:
            refined_question = base_question

        return refined_question

    def add_user_response(self, member_name: str, response: str):
        """Process and store the user response along with internal analysis."""
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": response,
            "timestamp": datetime.utcnow()
        })

        # Create an analysis prompt for the response
        analysis_prompt = f"""
Analyze this response from {member_name}:
---
{response}
---
Provide:
1. Key points (tasks done or in progress)
2. Any blockers/impediments noted
3. Suggested action items/follow-ups
Please format your answer as a bullet list.
"""
        try:
            analysis_result = model.generate_content(analysis_prompt).text.strip()
            # Append the internal analysis as an assistant message.
            analysis_message = f"[Internal Analysis]\n{analysis_result}"
            self.conversation_history.append({
                "role": "assistant",
                "content": analysis_message,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            print(f"Error analyzing response: {e}")

    def generate_ai_response(self) -> str:
        """
        Generate a follow-up question for the current conversation step.
        (By design, we use our fixed question mapping so the assistant does not reveal underlying context.)
        """
        if not self.team_members_list:
            return "Standup complete."
        current_member = self.team_members_list[self.current_member_index]
        return self.generate_question(current_member, self.conversation_step)

    def add_assistant_response(self, response: str):
        """Store the assistant's response in conversation history."""
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow()
        })

    def check_response_completeness(self, member_name: str, response: str) -> bool:
        """
        Analyze the response to determine if it is complete.
        If the response is trivial (like 'nothing' or 'no'), consider it complete.
        Otherwise, use the LLM to analyze further.
        """
        normalized = response.strip().lower()
        if normalized in ["nothing", "nothing thank you", "no", "none"]:
            return True  # Treat these as complete responses

        prompt = f"""
        You are an AI Scrum Master. Analyze the following standup response from {member_name}:
        ---
        {response}
        ---
        Answer with a single word: "Complete" if the response adequately covers all key topics (updates, plans, blockers), or "Incomplete" if further follow-up is needed. Then, provide a brief explanation.
        """
        try:
            result = model.generate_content(prompt).text.strip()
            print("Completeness Analysis:", result)
            if result.lower().startswith("complete"):
                return True
        except Exception as e:
            print(f"Error checking response completeness: {e}")
        return False

    def generate_summary(self) -> str:
        """Generate a summary of the standup."""
        summary_prompt = f"""
Summarize the following standup conversation:
---
{self.conversation_history}
---
Include:
- Key updates per team member
- Identified blockers
- Action items/follow-ups
- Overall sprint progress
Format the summary in markdown.
"""
        try:
            return model.generate_content(summary_prompt).text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Could not generate summary."

# --------------------------------------------------------------------------------
# 5) Teams Bot Implementation
# --------------------------------------------------------------------------------
def store_conversation_reference(activity: Activity):
    """Store a conversation reference from an incoming activity."""
    conversation_reference = TurnContext.get_conversation_reference(activity)
    CONVERSATION_REFERENCES[conversation_reference.user.id] = conversation_reference
    return conversation_reference

async def send_proactive_message(conversation_reference: ConversationReference, text: str):
    """Send a proactive message to a user using a stored conversation reference."""
    async def send_message(turn_context):
        await turn_context.send_activity(text)
    
    await BOT_ADAPTER.continue_conversation(
        conversation_reference,
        send_message,
        BOT_APP_ID
    )

# --------------------------------------------------------------------------------
# 6) FastAPI Routes and Data Models
# --------------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartStandupRequest(BaseModel):
    board_id: int
    user_id: str

class SendMessageRequest(BaseModel):
    user_id: str
    message: str

class StandupState(BaseModel):
    user_id: str
    current_member_index: int
    conversation_step: int
    messages: List[Dict]
    standup_started: bool
    team_members_list: List[str]
    nothing_count: int

standup_sessions: Dict[str, AIScrumMaster] = {}

# Teams Bot webhook endpoint
@app.post("/api/messages")
async def messages(req: Request):
    body = await req.json()
    activity = Activity().deserialize(body)
    
    # Store the conversation reference for later use
    conversation_reference = store_conversation_reference(activity)
    
    async def turn_handler(turn_context: TurnContext):
        # Only process message activities
        if turn_context.activity.type == ActivityTypes.message:
            # Get the message text
            text = turn_context.activity.text.strip()
            user_id = turn_context.activity.from_property.id
            user_name = turn_context.activity.from_property.name
            
            # Store the Teams user
            store_teams_user(user_id, user_name)
            
            # Process commands
            if text.lower().startswith("start standup"):
                # Extract board ID if provided, otherwise use default
                parts = text.split()
                board_id = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                
                # Initialize scrum master
                scrum_master = AIScrumMaster(user_id)
                scrum_master.conversation_reference = conversation_reference
                standup_sessions[user_id] = scrum_master
                
                if scrum_master.initialize_sprint_data(board_id):
                    await turn_context.send_activity(f"Standup started for board {board_id}!")
                    # Send the first question
                    if scrum_master.team_members_list:
                        member = scrum_master.team_members_list[0]
                        question = scrum_master.generate_question(member, 1)
                        await turn_context.send_activity(f"📋 **Current Member**: {member}\n\n{question}")
                    else:
                        await turn_context.send_activity("No team members found in the current sprint.")
                else:
                    await turn_context.send_activity("Could not find an active sprint for this board.")
            
            elif text.lower() == "get boards":
                try:
                    boards = get_boards()
                    boards_list = "\n".join([f"- Board ID: {b['id']}, Name: {b['name']}" for b in boards])
                    await turn_context.send_activity(f"Available boards:\n{boards_list}")
                except Exception as e:
                    await turn_context.send_activity(f"Error fetching boards: {str(e)}")
            
            elif text.lower() == "get summary":
                scrum_master = standup_sessions.get(user_id)
                if scrum_master and scrum_master.standup_started:
                    summary = scrum_master.generate_summary()
                    await turn_context.send_activity(f"# Standup Summary\n\n{summary}")
                    
                    # Store conversation and clean up
                    blockers = [msg['content'] for msg in scrum_master.conversation_history 
                               if "blocker" in msg['content'].lower()]
                    action_items = [msg['content'] for msg in scrum_master.conversation_history 
                                   if "action item" in msg['content'].lower()]
                    conversation_doc = {
                        "user_id": user_id,
                        "messages": scrum_master.conversation_history,
                        "blockers": blockers,
                        "action_items": action_items,
                        "summary": summary
                    }
                    store_conversation(conversation_doc)
                    scrum_master.standup_started = False
                    del standup_sessions[user_id]
                else:
                    await turn_context.send_activity("No active standup session found.")
            
            elif text.lower() == "help":
                help_text = """
# AI Scrum Master Bot Commands

- **start standup [board_id]** - Begin a new standup session
- **get boards** - List available JIRA boards
- **get summary** - Generate a summary of the current standup
- **help** - Show this help message

During a standup, simply respond to the bot's questions normally.
                """
                await turn_context.send_activity(help_text)
            
            else:
                # Process as a response to an ongoing standup
                scrum_master = standup_sessions.get(user_id)
                if scrum_master and scrum_master.standup_started:
                    current_member = scrum_master.team_members_list[scrum_master.current_member_index]
                    
                    # Process the user's response
                    scrum_master.add_user_response(current_member, text)
                    
                    # Check if response is complete
                    is_complete = scrum_master.check_response_completeness(current_member, text)
                    
                    if is_complete or text.lower() in ["nothing", "no", "none"]:
                        # Move to next member or next question
                        if scrum_master.conversation_step >= 4 or text.lower() in ["nothing", "no", "none"]:
                            # Thank the current member and move to the next
                            await turn_context.send_activity(f"Thanks for the update, {current_member}!")
                            scrum_master.current_member_index += 1
                            scrum_master.conversation_step = 1
                            
                            # Check if we have more members
                            if scrum_master.current_member_index < len(scrum_master.team_members_list):
                                next_member = scrum_master.team_members_list[scrum_master.current_member_index]
                                next_question = scrum_master.generate_question(next_member, 1)
                                await turn_context.send_activity(f"📋 **Current Member**: {next_member}\n\n{next_question}")
                            else:
                                # Standup complete
                                await turn_context.send_activity("All team members have completed their updates! Use 'get summary' to generate a standup summary.")
                        else:
                            # Move to the next question for the current member
                            scrum_master.conversation_step += 1
                            next_question = scrum_master.generate_question(
                                current_member, scrum_master.conversation_step
                            )
                            await turn_context.send_activity(next_question)
                    else:
                        # Ask a follow-up question
                        scrum_master.conversation_step += 1
                        next_question = scrum_master.generate_question(
                            current_member, scrum_master.conversation_step
                        )
                        await turn_context.send_activity(next_question)
                else:
                    # No active standup
                    await turn_context.send_activity("No active standup session. Use 'start standup [board_id]' to begin or 'help' for more commands.")
        
    # Process the incoming activity
    await BOT_ADAPTER.process_activity(activity, "", turn_handler)
    return {}

# Original REST API endpoints for non-Teams clients
@app.post("/start_standup/")
async def start_standup(request: StartStandupRequest):
    user_id = request.user_id
    board_id = request.board_id
    if user_id not in standup_sessions:
        standup_sessions[user_id] = AIScrumMaster(user_id)
    scrum_master = standup_sessions[user_id]
    if scrum_master.initialize_sprint_data(board_id):
        return {"message": "Standup started", "team_members": scrum_master.team_members_list}
    else:
        raise HTTPException(status_code=400, detail="No active sprint found for this board.")

@app.get("/get_next_question/{user_id}")
async def get_next_question(user_id: str):
    scrum_master = standup_sessions.get(user_id)
    if not scrum_master or not scrum_master.standup_started:
        raise HTTPException(status_code=400, detail="Standup not started or user not found.")

    if scrum_master.current_member_index < len(scrum_master.team_members_list):
        member = scrum_master.team_members_list[scrum_master.current_member_index]
        question = scrum_master.generate_question(member, scrum_master.conversation_step)
        scrum_master.messages.append({"role": "assistant", "content": question})
        scrum_master.add_assistant_response(question)
        return {"question": question, "current_member": member}
    else:
        return {"message": "Standup with all members complete. Use /get_summary."}

@app.post("/send_message/")
async def send_message(request: SendMessageRequest):
    user_id = request.user_id
    message = request.message
    scrum_master = standup_sessions.get(user_id)
    if not scrum_master or not scrum_master.standup_started:
        raise HTTPException(status_code=400, detail="Standup not started or user not found.")

    if scrum_master.current_member_index < len(scrum_master.team_members_list):
        member = scrum_master.team_members_list[scrum_master.current_member_index]
        scrum_master.messages.append({"role": "user", "content": message})
        scrum_master.add_user_response(member, message)
        
        is_complete = scrum_master.check_response_completeness(member, message)
        normalized_message = message.strip().lower()
        
        if normalized_message in ["nothing", "no", "none"]:
            # Increment nothing count
            scrum_master.nothing_count += 1
            
            # Move to next member
            scrum_master.current_member_index += 1
            scrum_master.conversation_step = 1
            
            if scrum_master.current_member_index < len(scrum_master.team_members_list):
                return {"message": "Response recorded. Moving to next team member."}
            else:
                return {"message": "All team members complete. Use /get_summary to get the standup summary."}
        
        if is_complete:
            if scrum_master.conversation_step >= 4:
                # All questions completed for this member, move to next member
                scrum_master.current_member_index += 1
                scrum_master.conversation_step = 1
                
                if scrum_master.current_member_index < len(scrum_master.team_members_list):
                    return {"message": "Response recorded. Moving to next team member."}
                else:
                    return {"message": "All team members complete. Use /get_summary to get the standup summary."}
            else:
                # Move to next question for current member
                scrum_master.conversation_step += 1
                return {"message": "Response recorded. Moving to next question."}
        else:
            # Need more information, stay on the same question
            return {"message": "Response recorded. Please provide more details."}

@app.get("/get_summary/{user_id}")
async def get_summary(user_id: str):
    scrum_master = standup_sessions.get(user_id)
    if not scrum_master:
        raise HTTPException(status_code=400, detail="User not found or standup not started.")
    
    summary = scrum_master.generate_summary()
    
    # Store the complete conversation in MongoDB
    blockers = [msg['content'] for msg in scrum_master.conversation_history 
               if "blocker" in msg['content'].lower()]
    action_items = [msg['content'] for msg in scrum_master.conversation_history 
                   if "action item" in msg['content'].lower()]
    
    conversation_doc = {
        "user_id": user_id,
        "messages": scrum_master.conversation_history,
        "blockers": blockers,
        "action_items": action_items,
        "summary": summary
    }
    store_conversation(conversation_doc)
    
    # Clean up the session
    standup_sessions[user_id].standup_started = False
    
    return {"summary": summary}

@app.get("/get_boards/")
async def fetch_jira_boards():
    try:
        boards = get_boards()
        return {"boards": boards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_sprints/{board_id}")
async def fetch_jira_sprints(board_id: int):
    try:
        sprints = fetch_sprint_details(board_id)
        return {"sprints": sprints}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_standup_state/{user_id}")
async def get_standup_state(user_id: str):
    scrum_master = standup_sessions.get(user_id)
    if not scrum_master:
        raise HTTPException(status_code=400, detail="User not found or standup not started.")
    
    return StandupState(
        user_id=user_id,
        current_member_index=scrum_master.current_member_index,
        conversation_step=scrum_master.conversation_step,
        messages=scrum_master.messages,
        standup_started=scrum_master.standup_started,
        team_members_list=scrum_master.team_members_list,
        nothing_count=scrum_master.nothing_count
    )

@app.post("/schedule_standup/")
async def schedule_standup(board_id: int, user_id: str, time: str):
    """Schedule a standup for a specific time (format: HH:MM)."""
    # This would likely use a task scheduler like APScheduler or Celery
    # For demonstration, we'll just validate inputs
    try:
        hours, minutes = map(int, time.split(':'))
        if not (0 <= hours < 24 and 0 <= minutes < 60):
            raise ValueError("Invalid time format")
        
        # Here you would add the scheduling logic
        return {"message": f"Standup scheduled for {time} for board {board_id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scheduling standup: {str(e)}")

@app.get("/get_historical_standups/{user_id}")
async def get_historical_standups(user_id: str, limit: int = 5):
    """Retrieve historical standup summaries for a user."""
    try:
        standups = get_previous_standups(user_id, limit)
        return {"historical_standups": standups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving historical standups: {str(e)}")

@app.post("/clear_standup_session/{user_id}")
async def clear_standup_session(user_id: str):
    """Clear a user's active standup session."""
    if user_id in standup_sessions:
        del standup_sessions[user_id]
        return {"message": "Standup session cleared successfully."}
    else:
        raise HTTPException(status_code=404, detail="No active standup session found for this user.")

# --------------------------------------------------------------------------------
# 7) Main Application Entry Point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Create indexes for MongoDB collections
    try:
        boards_collection.create_index("board_id", unique=True)
        sprints_collection.create_index("sprint_id", unique=True)
        issues_collection.create_index("issue_id", unique=True)
        users_collection.create_index("user_id", unique=True)
        teams_users_collection.create_index("teams_user_id", unique=True)
    except Exception as e:
        print(f"Error creating MongoDB indexes: {e}")
    
    print("Starting AI Scrum Master Application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    