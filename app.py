"""
LeaseGuard - Lease Agreement Review Assistant
Track 5: Legal Operations
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="LeaseGuard",
    description="Clause-by-clause lease agreement review assistant against standard company positions",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LeaseGuard"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
