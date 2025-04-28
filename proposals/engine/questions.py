"""
Question Generator Module

Dynamically generates follow-up questions based on existing answers.
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

from proposals.storage.session import SessionStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard proposal questions to ensure all necessary info is collected
STANDARD_QUESTIONS = [
    {
        "id": "project_name",
        "question": "What is the name of the project?",
        "question_type": "GENERAL",
        "importance": 10,
    },
    {
        "id": "project_goals",
        "question": "What are the main goals and objectives of this project?",
        "question_type": "GENERAL",
        "importance": 9,
    },
    {
        "id": "budget",
        "question": "What is the estimated budget for this project?",
        "question_type": "BUDGET",
        "importance": 8,
    },
    {
        "id": "timeline",
        "question": "What is the desired timeline or deadline for this project?",
        "question_type": "TIMELINE",
        "importance": 8,
    },
    {
        "id": "stakeholders",
        "question": "Who are the key stakeholders for this project?",
        "question_type": "GENERAL",
        "importance": 7,
    },
    {
        "id": "success_criteria",
        "question": "What are the success criteria for this project?",
        "question_type": "GENERAL",
        "importance": 7,
    },
    {
        "id": "scope",
        "question": "What is the scope of work for this project?",
        "question_type": "GENERAL",
        "importance": 9,
    },
]


class QuestionGenerator:
    """
    Dynamically generates and prioritizes questions for proposal creation
    """

    @staticmethod
    def get_standard_questions() -> List[Dict[str, Any]]:
        """
        Get the list of standard questions

        Returns:
            List[Dict]: List of standard question dictionaries
        """
        return STANDARD_QUESTIONS

    @staticmethod
    async def get_unanswered_questions(session_id: UUID) -> List[Dict[str, Any]]:
        """
        Get unanswered standard questions for a session

        Args:
            session_id: UUID of the session

        Returns:
            List[Dict]: List of unanswered standard questions
        """
        # Get existing answers
        storage = SessionStorage(session_id)
        answers = await storage.get_answers()

        # Filter out questions that have been answered
        unanswered = []
        for q in STANDARD_QUESTIONS:
            # Check if this question has been answered (simple substring match)
            answered = False
            for existing_q in answers.keys():
                # If the existing question contains this standard question, consider it answered
                if q["question"].lower() in existing_q.lower():
                    answered = True
                    break

            if not answered:
                unanswered.append(q)

        # Sort by importance
        unanswered.sort(key=lambda x: x["importance"], reverse=True)
        return unanswered

    @classmethod
    async def next(cls, session_id: UUID) -> Optional[str]:
        """
        Get the next most important question to ask

        Args:
            session_id: UUID of the session

        Returns:
            str or None: The next question to ask, or None if all standard questions answered
        """
        unanswered = await cls.get_unanswered_questions(session_id)
        if not unanswered:
            # If all standard questions are answered, use LLM to generate contextual question
            return await cls.generate_contextual_question(session_id)

        # Return the highest priority unanswered question
        return unanswered[0]["question"]

    @staticmethod
    async def generate_contextual_question(session_id: UUID) -> Optional[str]:
        """
        Use LLM to generate a contextual follow-up question based on existing answers

        Args:
            session_id: UUID of the session

        Returns:
            str or None: A contextual follow-up question or None if LLM not available
        """
        # Get the LLM
        llm = Settings.llm
        if not llm:
            logger.warning("No LLM available for generating contextual questions")
            return None

        try:
            # Get existing answers as chat history
            storage = SessionStorage(session_id)
            history = await storage.get_chat_history()

            # Create chat messages
            messages = [
                ChatMessage(
                    role="system",
                    content="""
                You are a proposal specialist helping gather information for a project proposal.
                Based on the previous questions and answers, identify the most important missing information
                and ask ONE specific follow-up question to help create a comprehensive proposal.
                Focus on gaps in: scope details, budget clarification, timeline specifics, requirements,
                key deliverables, or success criteria.
                """,
                ),
            ]

            # Add existing chat history
            for msg in history:
                messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

            # Add final user instruction
            messages.append(
                ChatMessage(
                    role="user",
                    content="Based on our conversation so far, what's the most important question I should answer next for the proposal?",
                )
            )

            # Generate the next question
            response = llm.chat(messages)

            # Extract the question from the response
            question = response.content.strip()

            # If it's a very long response, try to extract just the question
            if len(question) > 200:
                # Look for question marks or common question patterns
                sentences = question.split(".")
                for sentence in sentences:
                    if "?" in sentence:
                        return sentence.strip()

                # If no clear question found, use the first 200 chars
                return question[:200] + "..."

            return question

        except Exception as e:
            logger.error(f"Error generating contextual question: {e}")
            return "Is there any additional information you would like to provide for the proposal?"
