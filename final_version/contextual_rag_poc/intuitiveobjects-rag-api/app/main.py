from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.db.mongodb import connect_to_mongodb, close_mongodb_connection
from app.api.v1.routes import router as v1_router
from app.api.v2.routes import router as v2_router


from app.utils.auth import get_current_user
from app.core.exception_handlers import (
    http_exception_handler,
    validation_exception_handler,
)

app = FastAPI(
    title="Demo API",
    description="This is a Demo API for testing purposes",
    version="1.0.0",
)



# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Setup MongoDB connection events

app.add_event_handler("startup", connect_to_mongodb)
app.add_event_handler("shutdown", close_mongodb_connection)

# Include API routes
app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")

from app.pipeline.chroma_db import init_chroma_collection

@app.on_event("startup")
async def startup_chroma():
     await init_chroma_collection()
       

@app.get("/")
async def root():
    return {"message": "Welcome to Demo API with MongoDB"}
