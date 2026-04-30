from fastapi import FastAPI

app = FastAPI(
    title="ThermoPINN API",
    description="Physics-informed thermal prediction API for engineering systems.",
    version="0.1.0",
)

@app.get("/")
def root():
    return {
        "message": "ThermoPINN API",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }
