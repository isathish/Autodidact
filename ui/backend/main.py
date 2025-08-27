from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Project Autodidact API")

# Example in-memory state
agent_state = {
    "current_goal": None,
    "rewards": [],
    "knowledge": []
}

class GoalRequest(BaseModel):
    goal: str

@app.get("/status")
def get_status():
    return {
        "current_goal": agent_state["current_goal"],
        "recent_rewards": agent_state["rewards"][-10:],
        "knowledge_count": len(agent_state["knowledge"])
    }

@app.post("/send_goal")
def send_goal(request: GoalRequest):
    agent_state["current_goal"] = request.goal
    # TODO: Integrate with goals.generator to push into agent
    return {"message": f"Goal '{request.goal}' set successfully."}

@app.get("/get_knowledge")
def get_knowledge():
    return {"knowledge": agent_state["knowledge"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
