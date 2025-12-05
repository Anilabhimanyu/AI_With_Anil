from fastapi import FastAPI

app = FastAPI() # ASGI application server

@app.get("/")
async def root():
    return {"message": "FastAPI To Do App"}
