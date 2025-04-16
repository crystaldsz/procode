import os
from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import google.generativeai as genai
import requests
from requests.auth import HTTPBasicAuth
import re
import json
import pandas as pd

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

# MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["jira_db"]
boards_collection = db["boards"]
sprints_collection = db["sprints"]
issues_collection = db["issues"]
users_collection = db["users"]
conversations_collection = db["conversations"]

# MongoDB Helper Functions
def store_board(board: Dict):
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
    conversation_doc["date"] = datetime.utcnow()
    conversations_collection.insert_one(conversation_doc)

def get_previous_standups(user_id: str, limit=5):
    cursor = conversations_collection.find({"user_id": user_id}).sort("date", -1).limit(limit)
    return list(cursor)

# JIRA Integration Functions
def extract_content_from_adf(content):
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
# AIScrumMaster Class
class AIScrumMaster:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history = []
        self.current_sprint = None
        self.team_members = set()
        self.blockers = []
        self.action_items = []
        self.context_cache = {}
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
        previous_standups = get_previous_standups(self.user_id, limit=3)
        for doc in reversed(previous_standups):
            self.conversation_history.extend(doc.get("messages", []))

    def initialize_sprint_data(self, board_id: int):
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
                return True
        return False

    def get_member_tasks(self, member_name: str) -> List[Dict]:
        if not self.current_sprint:
            return []
        return [issue for issue in self.current_sprint['issues'] if issue.get('Assignee') == member_name]

    def build_tasks_context(self, member_name: str) -> str:
        tasks = self.get_member_tasks(member_name)
        if not tasks:
            return "No tasks assigned currently."
        return "\n".join([f"- {task['Key']}: {task['Summary']} (Status: {task['Status']})" for task in tasks])

    def get_mongo_context(self, member_name: str) -> str:
        docs = get_previous_standups(self.user_id, limit=5)
        context_lines = []
        for doc in docs:
            for msg in doc.get("messages", []):
                if member_name in msg.get("content", "") or msg.get("role") == "assistant":
                    context_lines.append(msg["content"])
        if context_lines:
            return "\nRelevant Historical Updates:\n" + "\n".join(f"- {line}" for line in context_lines)
        return "No historical updates available."

    def get_contextual_history(self, member_name: str) -> str:
        return self.get_mongo_context(member_name)

    def generate_question(self, member_name: str, step: int) -> str:
        base_questions = {
            1: f"Hey {member_name}, how are you doing today? How are you feeling about your tasks?",
            2: f"Could you update me on what you accomplished recently, and if you ran into any challenges?",
            3: f"Great, thanks for the update! What's on your agenda for today?",
            4: f"Are there any blockers or issues that you need help with?",
            5: f"Anything else you'd like to add before we wrap up?"
        }
        base_question = base_questions.get(step, "Is there anything else you'd like to discuss?")
        tasks_context = self.build_tasks_context(member_name)
        mongo_context = self.get_mongo_context(member_name)
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
        refined_question = model.generate_content(prompt).text.strip()
        if not refined_question:
            refined_question = base_question
        return refined_question

    def add_user_response(self, member_name: str, response: str):
        self.conversation_history.append({
            "role": "user",
            "content": response,
            "timestamp": datetime.utcnow()
        })
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
        analysis_result = model.generate_content(analysis_prompt).text.strip()
        analysis_message = f"[Internal Analysis]\n{analysis_result}"
        self.conversation_history.append({
            "role": "assistant",
            "content": analysis_message,
            "timestamp": datetime.utcnow()
        })

    def generate_ai_response(self, current_member_index, conversation_step) -> str:
        current_member = list(self.team_members)[current_member_index]
        return self.generate_question(current_member, conversation_step)

    def add_assistant_response(self, response: str):
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow()
        })

    def check_response_completeness(self, member_name: str, response: str) -> bool:
        normalized = response.strip().lower()
        if normalized in ["nothing", "nothing thank you", "no", "none"]:
            return True
        prompt = f"""
        You are an AI Scrum Master. Analyze the following standup response from {member_name}:
        ---
        {response}
        ---
        Answer with a single word: "Complete" if the response adequately covers all key topics (updates, plans, blockers), or "Incomplete" if further follow-up is needed. Then, provide a brief explanation.
        """
        result = model.generate_content(prompt).text.strip()
        print("Completeness Analysis:", result)
        if result.lower().startswith("complete"):
            return True
        return False

    def generate_summary(self) -> str:
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
        return model.generate_content(summary_prompt).text.strip()

app = FastAPI() #<----------------- This is the missing line.

class StartStandupRequest(BaseModel):
    board_id: int
    user_id:str

class UserResponse(BaseModel):
    user_id: str
    response: str
    member_name:str
    current_member_index:int
    conversation_step:int

class EndStandupRequest(BaseModel):
    user_id:str
    summary:str

standup_sessions:Dict[str, AIScrumMaster]={}

@app.post("/start_standup/")
async def start_standup(request_data: StartStandupRequest):
    user_id = request_data.user_id
    board_id = request_data.board_id
    scrum_master = AIScrumMaster(user_id)
    if scrum_master.initialize_sprint_data(board_id):
        standup_sessions[user_id]=scrum_master
        return {"message": "Standup started.","team_members":list(scrum_master.team_members)}
    else:
        raise HTTPException(status_code=400, detail="No active sprint found.")

@app.post("/user_response/")
async def user_response(response_data: UserResponse):
    user_id = response_data.user_id
    response = response_data.response
    member_name = response_data.member_name
    current_member_index = response_data.current_member_index
    conversation_step = response_data.conversation_step

    scrum_master = standup_sessions.get(user_id)
    if not scrum_master:
        raise HTTPException(status_code=404, detail="Standup session not found.")

    scrum_master.add_user_response(member_name, response)
    is_complete = scrum_master.check_response_completeness(member_name, response)
    if is_complete:
        return {"message": "Response processed.","complete":True}
    else:
        next_question = scrum_master.generate_ai_response(current_member_index, conversation_step+1)
        scrum_master.add_assistant_response(next_question)
        return {"message": "Response processed.","complete":False, "next_question":next_question}

@app.get("/generate_question/{user_id}/{current_member_index}/{step}")
async def generate_question(user_id: str, current_member_index: int, step: int):
    scrum_master = standup_sessions.get(user_id)
    if not scrum_master:
        raise HTTPException(status_code=404, detail="Standup session not found.")
    question = scrum_master.generate_ai_response(current_member_index, step)
    scrum_master.add_assistant_response(question)
    return {"question": question}

@app.post("/end_standup/")
async def end_standup(request_data: EndStandupRequest):
    user_id = request_data.user_id
    summary = request_data.summary

    scrum_master = standup_sessions.get(user_id)
    if not scrum_master:
        raise HTTPException(status_code=404, detail="Standup session not found.")

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
    del standup_sessions[user_id]
    return {"message": "Standup ended. Have a great day!"}

@app.get("/boards/")
async def get_boards_endpoint():
    return get_boards()

@app.get("/sprints/{board_id}")
async def get_sprints_endpoint(board_id: int):
    return fetch_sprint_details(board_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)