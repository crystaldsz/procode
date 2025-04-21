import os
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import google.generativeai as genai
import requests
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
# 5) FastAPI Routes and Data Models
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

        # Check if the response is trivial
        if message.strip().lower() in ["nothing", "nothing thank you", "no", "none"]:
            scrum_master.nothing_count += 1
        else:
            scrum_master.nothing_count = 0  # Reset if the response is meaningful

        # Use the completeness checker
        is_complete = scrum_master.check_response_completeness(member, message)

        # If the response is complete or repeated "nothing" responses are detected
        if is_complete or scrum_master.nothing_count >= 2:
            final_message = f"Thanks for the update, {member}. I'll now move on to the next team member."
            scrum_master.messages.append({"role": "assistant", "content": final_message})
            scrum_master.add_assistant_response(final_message)
            scrum_master.current_member_index += 1
            scrum_master.conversation_step = 1
            scrum_master.messages = []  # Reset messages for the next member
            scrum_master.nothing_count = 0  # Reset the counter
            if scrum_master.current_member_index < len(scrum_master.team_members_list):
                next_member = scrum_master.team_members_list[scrum_master.current_member_index]
                next_question = scrum_master.generate_question(next_member, scrum_master.conversation_step)
                scrum_master.messages.append({"role": "assistant", "content": next_question})
                scrum_master.add_assistant_response(next_question)
                return {"response": "Acknowledged, moving to next member.", "next_question": next_question, "current_member": next_member}
            else:
                return {"response": "Acknowledged. Standup with all members complete. Use /get_summary."}
        else:
            scrum_master.conversation_step += 1
            next_question = scrum_master.generate_question(member, scrum_master.conversation_step)
            scrum_master.messages.append({"role": "assistant", "content": next_question})
            scrum_master.add_assistant_response(next_question)
            return {"response": "Acknowledged.", "next_question": next_question}
    else:
        return {"message": "Standup with all members complete. Use /get_summary."}

@app.get("/get_summary/{user_id}")
async def get_summary(user_id: str):
    scrum_master = standup_sessions.get(user_id)
    if not scrum_master:
        raise HTTPException(status_code=404, detail="User session not found.")
    summary = scrum_master.generate_summary()
    blockers = [msg['content'] for msg in scrum_master.conversation_history if "blocker" in msg['content'].lower()]
    action_items = [msg['content'] for msg in scrum_master.conversation_history if "action item" in msg['content'].lower()]
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
    return {"summary": summary, "blockers": blockers, "action_items": action_items, "message": "Standup ended."}

@app.get("/get_boards/")
async def fetch_jira_boards():
    return get_boards()

@app.get("/get_sprints/{board_id}")
async def fetch_jira_sprints(board_id: int):
    return fetch_sprint_details(board_id)
                