from fastapi import FastAPI

app = FastAPI(
    title="FYP RAG Knowledge Assistant API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "FYP RAG Knowledge Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }