"""
Session Storage for Q&A Sessions

This module provides the storage layer for managing sessions and answers in the
proposal generation system.
"""

import json
import logging
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class User(BaseModel):
    """User model for session tracking"""

    id: str
    name: str
    email: Optional[str] = None


class Answer(BaseModel):
    """Model for storing question and answer pairs"""

    question: str
    answer: str
    question_type: str = "GENERAL"  # GENERAL, BUDGET, TIMELINE, etc.
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Represents a Q&A session for proposal generation"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user: User
    answers: Dict[str, Answer] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionManager:
    """
    Manages proposal generation sessions with PostgreSQL persistent storage
    """

    # Fallback in-memory store if database connection fails
    _sessions: Dict[str, Session] = {}
    _connection_pool = None
    _initialized = False

    @classmethod
    async def initialize(cls):
        """Initialize the database connection pool and create tables if needed"""
        if cls._initialized:
            return

        db_connection = os.environ.get(
            "DB_CONNECTION"
            )

        try:
            # Create connection pool
            cls._connection_pool = await asyncpg.create_pool(db_connection)

            # Create tables if they don't exist
            async with cls._connection_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        user_data JSONB NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS answers (
                        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        question_type TEXT DEFAULT 'GENERAL',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        PRIMARY KEY (session_id, question)
                    )
                """)

            logger.info("Database connection established and tables created")
            cls._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            logger.warning("Falling back to in-memory storage")

    @classmethod
    async def create(cls, user: User) -> Session:
        """
        Create a new session for a user

        Args:
            user: User for whom to create a session

        Returns:
            Session: The newly created session
        """
        await cls.initialize()

        session = Session(user=user)

        if cls._connection_pool:
            try:
                async with cls._connection_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO sessions (id, user_data, created_at, updated_at, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        session.id,
                        json.dumps(user.model_dump()),
                        session.created_at,
                        session.updated_at,
                        json.dumps(session.metadata),
                    )
                logger.info(
                    f"Created session {session.id} for user {user.name} in database"
                )
            except Exception as e:
                logger.error(f"Failed to create session in database: {e}")
                # Fallback to in-memory
                cls._sessions[session.id] = session
                logger.info(
                    f"Created session {session.id} for user {user.name} in memory"
                )
        else:
            # Use in-memory storage
            cls._sessions[session.id] = session
            logger.info(f"Created session {session.id} for user {user.name} in memory")

        return session

    @classmethod
    async def get(cls, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID

        Args:
            session_id: ID of the session to retrieve

        Returns:
            Session or None if not found
        """
        await cls.initialize()

        if cls._connection_pool:
            try:
                async with cls._connection_pool.acquire() as conn:
                    # Get session data
                    session_record = await conn.fetchrow(
                        """
                        SELECT id, user_data, created_at, updated_at, metadata
                        FROM sessions
                        WHERE id = $1
                        """,
                        session_id,
                    )

                    if not session_record:
                        logger.warning(f"Session {session_id} not found in database")
                        return None

                    # Get answers for this session
                    answer_records = await conn.fetch(
                        """
                        SELECT question, answer, question_type, created_at, metadata
                        FROM answers
                        WHERE session_id = $1
                        """,
                        session_id,
                    )

                    # Build session object
                    user_data = json.loads(session_record["user_data"])
                    user = User(**user_data)

                    session = Session(
                        id=session_record["id"],
                        user=user,
                        created_at=session_record["created_at"],
                        updated_at=session_record["updated_at"],
                        metadata=json.loads(session_record["metadata"]),
                    )

                    # Add answers
                    for record in answer_records:
                        answer = Answer(
                            question=record["question"],
                            answer=record["answer"],
                            question_type=record["question_type"],
                            created_at=record["created_at"],
                            metadata=json.loads(record["metadata"]),
                        )
                        session.answers[answer.question] = answer

                    return session
            except Exception as e:
                logger.error(f"Failed to get session from database: {e}")
                # Fallback to in-memory
                return cls._sessions.get(session_id)
        else:
            # Use in-memory storage
            session = cls._sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found in memory")
            return session

    @classmethod
    async def add_answer(cls, session_id: str, answer: Answer) -> bool:
        """
        Add or update an answer in a session

        Args:
            session_id: ID of the session
            answer: Answer to add/update

        Returns:
            bool: True if successful, False otherwise
        """
        await cls.initialize()

        if cls._connection_pool:
            try:
                async with cls._connection_pool.acquire() as conn:
                    # Check if session exists
                    session_exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = $1)",
                        session_id,
                    )

                    if not session_exists:
                        logger.warning(f"Session {session_id} not found in database")
                        return False

                    # Update session's updated_at timestamp
                    await conn.execute(
                        """
                        UPDATE sessions 
                        SET updated_at = $1
                        WHERE id = $2
                        """,
                        datetime.now(),
                        session_id,
                    )

                    # Insert or update the answer
                    await conn.execute(
                        """
                        INSERT INTO answers (session_id, question, answer, question_type, created_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (session_id, question) 
                        DO UPDATE SET 
                            answer = EXCLUDED.answer,
                            question_type = EXCLUDED.question_type,
                            metadata = EXCLUDED.metadata
                        """,
                        session_id,
                        answer.question,
                        answer.answer,
                        answer.question_type,
                        answer.created_at,
                        json.dumps(answer.metadata),
                    )

                logger.info(
                    f"Added answer to question '{answer.question}' in session {session_id} in database"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to add answer in database: {e}")
                # Fallback to in-memory
                session = cls._sessions.get(session_id)
                if not session:
                    return False
                session.answers[answer.question] = answer
                session.updated_at = datetime.now()
                logger.info(
                    f"Added answer to question '{answer.question}' in session {session_id} in memory"
                )
                return True
        else:
            # Use in-memory storage
            session = cls._sessions.get(session_id)
            if not session:
                return False
            session.answers[answer.question] = answer
            session.updated_at = datetime.now()
            logger.info(
                f"Added answer to question '{answer.question}' in session {session_id} in memory"
            )
            return True

    @classmethod
    async def get_answers(cls, session_id: str) -> Dict[str, Answer]:
        """
        Get all answers for a session

        Args:
            session_id: ID of the session

        Returns:
            Dict[str, Answer]: Dictionary of question-answer pairs
        """
        session = await cls.get(session_id)
        if not session:
            return {}
        return session.answers

    @classmethod
    async def to_dict(cls, session_id: str) -> Dict[str, Any]:
        """
        Convert a session to a dictionary

        Args:
            session_id: ID of the session

        Returns:
            Dict: Session as a dictionary
        """
        session = await cls.get(session_id)
        if not session:
            return {}
        return session.model_dump()


class SessionStorage:
    """
    Wrapper for accessing session storage

    This class provides a more convenient interface for working with sessions
    and supports both in-memory and persistent PostgreSQL storage.
    """

    def __init__(self, session_id: str):
        """
        Initialize storage for a specific session

        Args:
            session_id: ID of the session to work with
        """
        self.session_id = session_id

    async def get_answers(self) -> Dict[str, str]:
        """
        Get answers in a simple dict format

        Returns:
            Dict[str, str]: Dictionary mapping questions to answers
        """
        answers = await SessionManager.get_answers(self.session_id)
        return {q: a.answer for q, a in answers.items()}

    async def get_answer(self, question: str) -> Optional[str]:
        """
        Get a specific answer

        Args:
            question: The question to get the answer for

        Returns:
            str or None: The answer or None if not found
        """
        answers = await SessionManager.get_answers(self.session_id)
        answer_obj = answers.get(question)
        return answer_obj.answer if answer_obj else None

    async def upsert_answer(self, answer: Union[Answer, Dict]) -> bool:
        """
        Add or update an answer

        Args:
            answer: Answer object or dictionary with answer data

        Returns:
            bool: True if successful
        """
        if isinstance(answer, dict):
            if "question" not in answer or "answer" not in answer:
                logger.error("Answer dict must contain 'question' and 'answer' keys")
                return False
            answer = Answer(**answer)

        return await SessionManager.add_answer(self.session_id, answer)

    async def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get session answers formatted as chat history

        Returns:
            List[Dict[str, str]]: List of messages in chat format
        """
        answers = await SessionManager.get_answers(self.session_id)
        chat_history = []

        for question, answer_obj in answers.items():
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer_obj.answer})

        return chat_history
