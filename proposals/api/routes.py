"""
API Routes for Proposal Generator

This module defines the FastAPI routes for the proposal generator service.
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from proposals.storage.session import User, Answer, SessionManager, SessionStorage
from proposals.engine.generator import (
    ProposalFormat,
    generate_proposal,
    generate_proposal_streaming,
)
from proposals.engine.questions import QuestionGenerator
from proposals.validation.validator import AnswerValidator, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["proposal-generator"])


# Request/Response models
class CreateSessionRequest(BaseModel):
    """Request model for creating a new session"""

    user_id: str
    user_name: str
    user_email: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response model for session operations"""

    session_id: str
    message: str = "Session created successfully"


class AnswerRequest(BaseModel):
    """Request model for submitting an answer"""

    question: str
    answer: str
    question_type: str = "GENERAL"
    metadata: Optional[Dict[str, Any]] = None


class AnswerResponse(BaseModel):
    """Response model after processing an answer"""

    success: bool
    next_question: Optional[str] = None
    message: Optional[str] = None


class GenerateProposalRequest(BaseModel):
    """Request model for generating a proposal"""

    format: str = ProposalFormat.DETAILED
    include_metadata: bool = False


class GenerateProposalResponse(BaseModel):
    """Response model for proposal generation"""

    proposal: str
    format: str
    metadata: Optional[Dict[str, Any]] = None


# API endpoints
@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Initialize a stateful Q&A session"""
    try:
        user = User(
            id=request.user_id, name=request.user_name, email=request.user_email
        )
        session = await SessionManager.create(user)
        return {"session_id": session.id}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@router.post("/sessions/{session_id}/answers", response_model=AnswerResponse)
async def process_answer(session_id: str, answer_req: AnswerRequest):
    """Validate and index answers as structured data"""
    try:
        # Convert to Answer object
        answer = Answer(
            question=answer_req.question,
            answer=answer_req.answer,
            question_type=answer_req.question_type,
            metadata=answer_req.metadata or {},
        )

        # Validate the answer
        try:
            validated = AnswerValidator.validate(answer)
        except ValidationError as e:
            return {"success": False, "message": str(e)}

        # Store the answer
        storage = SessionStorage(session_id)
        success = await storage.upsert_answer(validated)

        if not success:
            return {
                "success": False,
                "message": f"Failed to store answer for session {session_id}",
            }

        # Get next question
        next_question = await QuestionGenerator.next(session_id)

        return {
            "success": True,
            "next_question": next_question,
            "message": "Answer processed successfully",
        }

    except Exception as e:
        logger.error(f"Error processing answer: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process answer: {str(e)}"
        )


@router.post(
    "/sessions/{session_id}/proposals", response_model=GenerateProposalResponse
)
async def generate_proposal_endpoint(session_id: str, request: GenerateProposalRequest):
    """Execute the proposal generation pipeline"""
    try:
        # Generate proposal
        result = await generate_proposal(session_id=session_id, format=request.format)

        # Return result, optionally including metadata
        response = {"proposal": result["proposal"], "format": result["format"]}

        if request.include_metadata:
            response["metadata"] = result["metadata"]

        return response

    except Exception as e:
        logger.error(f"Error generating proposal: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate proposal: {str(e)}"
        )


@router.get("/sessions/{session_id}/proposals/stream")
async def stream_proposal(session_id: str, format: str = ProposalFormat.DETAILED):
    """Stream a proposal as it's being generated"""
    try:
        # Set up streaming generator
        streaming_gen = await generate_proposal_streaming(
            session_id=session_id, format=format
        )

        # Create event stream
        async def event_generator():
            try:
                async for token in streaming_gen:
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error setting up proposal stream: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stream proposal: {str(e)}"
        )


@router.get("/sessions/{session_id}/questions/next")
async def get_next_question(session_id: str):
    """Get the next question to ask"""
    try:
        next_question = await QuestionGenerator.next(session_id)
        return {"question": next_question}
    except Exception as e:
        logger.error(f"Error getting next question: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get next question: {str(e)}"
        )


@router.get("/formats")
async def get_proposal_formats():
    """Get available proposal formats"""
    return {"formats": ProposalFormat.values()}
