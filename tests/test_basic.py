"""
Basic tests for the proposal generator
"""

import unittest
import uuid
from unittest.mock import patch

from proposals.storage.session import User, Answer, SessionManager
from proposals.validation.validator import AnswerValidator
from proposals.engine.questions import QuestionGenerator


class TestProposalGenerator(unittest.TestCase):
    """Basic tests for the proposal generator"""

    def setUp(self):
        """Set up test data"""
        self.user = User(id="test_user", name="Test User")
        # Clear sessions before each test
        SessionManager._sessions = {}

    def test_session_creation(self):
        """Test session creation"""
        session = SessionManager.create(self.user)
        self.assertIsNotNone(session)
        self.assertEqual(session.user.id, "test_user")
        self.assertEqual(session.user.name, "Test User")

        # Test session retrieval
        retrieved = SessionManager.get(session.id)
        self.assertEqual(retrieved.id, session.id)

    def test_answer_validation(self):
        """Test answer validation"""
        # Valid answer
        answer = Answer(
            question="What is the project name?",
            answer="Test Project",
            question_type="GENERAL",
        )
        validated = AnswerValidator.validate(answer)
        self.assertIsNotNone(validated)

        # Test budget validation
        budget_answer = Answer(
            question="What is your budget?", answer="10000", question_type="BUDGET"
        )
        validated = AnswerValidator.validate(budget_answer)
        self.assertIsNotNone(validated)

    def test_question_generation(self):
        """Test question generation"""
        # Create a session
        session = SessionManager.create(self.user)

        # Get first question
        question = QuestionGenerator.next(session.id)
        self.assertIsNotNone(question)

        # Check that there are standard questions
        standard_questions = QuestionGenerator.get_standard_questions()
        self.assertTrue(len(standard_questions) > 0)

    @patch("proposals.engine.questions.QuestionGenerator.generate_contextual_question")
    def test_contextual_questions(self, mock_generate):
        """Test contextual question generation when all standard questions are answered"""
        mock_generate.return_value = (
            "What challenges do you anticipate in this project?"
        )

        # Create session
        session = SessionManager.create(self.user)

        # Answer all standard questions
        for q in QuestionGenerator.get_standard_questions():
            answer = Answer(
                question=q["question"],
                answer=f"Test answer for {q['id']}",
                question_type=q["question_type"],
            )
            SessionManager.add_answer(session.id, answer)

        # Now all standard questions are answered, should call the contextual method
        next_q = QuestionGenerator.next(session.id)
        self.assertEqual(next_q, "What challenges do you anticipate in this project?")
        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
