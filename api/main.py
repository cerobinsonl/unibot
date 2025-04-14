from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create a small delay to ensure all modules are properly loaded
import time
time.sleep(2)

# Import routers - these imports need to happen after any fixes to model files
from routers import chat, visualizations, websockets

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Connect to services, load models, etc.
    logger.info("Starting up FastAPI application")
    yield
    # Shutdown logic: Close connections, release resources
    logger.info("Shutting down FastAPI application")

# Create FastAPI app
app = FastAPI(
    title="University Administrative ChatBot API",
    description="API for interacting with the university's multi-agent system",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Django domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(visualizations.router)
app.include_router(websockets.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred"},
    )

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=os.getenv("API_DEBUG", "false").lower() == "true"
    )