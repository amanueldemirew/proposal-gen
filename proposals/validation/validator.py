"""
Answer Validation Module

This module validates user answers against business rules and handles LLM-based validation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from proposals.storage.session import Answer
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for validation errors"""

    pass


class AnswerValidator:
    """
    Validates answers against business rules and constraints
    """

    # Simple rule definitions - would be loaded from a database in production
    RULES = {
        "BUDGET": {
            "min_value": 0,
            "max_value": 10000000,  # 10M max budget
            "required": True,
        },
        "TIMELINE": {
            "min_days": 1,
            "max_days": 365 * 2,  # 2 years max
            "required": True,
        },
        "GENERAL": {"min_length": 3, "max_length": 5000, "required": False},
    }

    @staticmethod
    def validate(answer: Answer) -> Dict[str, Any]:
        """
        Validate an answer against business rules

        Args:
            answer: The answer to validate

        Returns:
            Dict: The validated answer as a dictionary

        Raises:
            ValidationError: If the answer fails validation
        """
        # Basic validation based on answer type
        answer_type = answer.question_type
        question = answer.question
        value = answer.answer

        # Get rule set for this answer type
        rules = AnswerValidator.RULES.get(answer_type, AnswerValidator.RULES["GENERAL"])

        # Apply type-specific validations
        if answer_type == "BUDGET":
            try:
                # Try to convert to float for budget values
                budget_value = float(value.replace(",", "").replace("$", "").strip())
                if budget_value < rules["min_value"]:
                    raise ValidationError(f"Budget must be positive")
                if budget_value > rules["max_value"]:
                    raise ValidationError(
                        f"Budget exceeds maximum allowed value of ${rules['max_value']:,.2f}"
                    )
            except ValueError:
                raise ValidationError("Budget must be a valid number")

        elif answer_type == "TIMELINE":
            # For timeline answers, we'd have more complex validation
            # This is a simplified version
            if len(value) < 5:  # Just a simple length check for demonstration
                raise ValidationError("Timeline description is too short")

        # Generic validations for all answer types
        if rules.get("required", False) and not value:
            raise ValidationError(f"Answer for '{question}' is required")

        if len(value) < rules.get("min_length", 0):
            raise ValidationError(
                f"Answer is too short, minimum {rules['min_length']} characters"
            )

        if len(value) > rules.get("max_length", 1000000):
            raise ValidationError(
                f"Answer is too long, maximum {rules['max_length']} characters"
            )

        # If we get here, basic validation passed
        # Convert to dictionary for storage
        return answer.model_dump()

    @staticmethod
    async def validate_with_llm(answer: Answer, llm=None) -> Tuple[bool, Optional[str]]:
        """
        Use LLM to validate answer quality and relevance

        Args:
            answer: The answer to validate
            llm: Optional LLM to use, defaults to Settings.llm

        Returns:
            Tuple[bool, str]: (is_valid, reason) - validity and explanation
        """
        # Use default LLM if none provided
        llm = llm or Settings.llm
        if not llm:
            logger.warning("No LLM available for validation, skipping LLM validation")
            return True, None

        try:
            # Use LlamaIndex's FaithfulnessEvaluator to check if the answer is reasonable
            evaluator = FaithfulnessEvaluator(llm=llm)

            # The evaluator expects a query and response
            # Here we're treating the question as the query and the answer as the response
            eval_result = evaluator.evaluate(
                query=answer.question, response=answer.answer
            )

            if not eval_result.passing:
                return False, eval_result.feedback

            return True, None

        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            # If LLM validation fails, we still accept the answer but log the error
            return True, f"LLM validation error: {str(e)}"
