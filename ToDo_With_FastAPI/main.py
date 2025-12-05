from fastapi import FastAPI

app = FastAPI(title="Todo API", version="1.0") # ASGI application server

@app.get("/")
async def root():
    """Root endpoint - returns welcome message."""
    return {"message": "Todo API is running", "version": "1.0"}

from fastapi import HTTPException

@app.get("/todos/{todo_id}")
async def read_todo(todo_id: int):
    """Read todo by ID (int validation automatic)."""
    if todo_id > 100:
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"todo_id": todo_id, "item": f"Buy milk (ID: {todo_id})"}

@app.get("/todos/{todo_id}/{priority}")
async def read_todo_priority(todo_id: int, priority: str):
    """Multiple path params with types."""
    return {
        "todo_id": todo_id,
        "priority": priority.upper(),
        "item": f"High priority task {todo_id}"
    }
