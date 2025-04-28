"""
Proposal Generator Engine

Core engine for generating proposals using LlamaIndex with Gemini.
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import SubQuestionQueryEngine

from proposals.storage.session import SessionStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Proposal format options
class ProposalFormat:
    BRIEF = "brief"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    FORMAL = "formal"

    @classmethod
    def values(cls) -> List[str]:
        return [cls.BRIEF, cls.DETAILED, cls.EXECUTIVE, cls.FORMAL]


# Templates for different proposal formats
PROPOSAL_TEMPLATES = {
    ProposalFormat.BRIEF: """
    Generate a comprehensive proposal with these requirements:
    {% for item in answers %}
    - {{item.question}}: {{item.answer}}
    {% endfor %}
    
    Include detailed sections for:
    1. Executive Summary (at least 150 words) - Provide a compelling overview of the proposal's key points
    2. Project Scope (at least 200 words) - Detail specific deliverables, services, and features
    3. Budget (at least 100 words) - Break down costs by category with justification
    4. Timeline (at least 150 words) - Include detailed milestones with dates
    5. Expected Outcomes (at least 150 words) - Describe measurable results

    Ensure the proposal is at least ONE FULL PAGE (minimum 800 words total). 
    Use professional language throughout and provide specific, actionable details rather than vague statements.
    """,
    ProposalFormat.DETAILED: """
    Generate a comprehensive proposal with these requirements:
    {% for item in answers %}
    - {{item.question}}: {{item.answer}}
    {% endfor %}
    
    Include detailed sections for (each section must be thorough and well-developed):
    1. Executive Summary (200+ words) - Compelling overview that captures key selling points
    2. Project Background & Goals (250+ words) - Thorough analysis of the situation and objectives
    3. Scope of Work (300+ words) - Comprehensive breakdown of all deliverables with specifics
    4. Detailed Budget (200+ words) - Line-item breakdown with justification for each cost
    5. Timeline with milestones (200+ words) - Detailed schedule with specific dates and dependencies
    6. Team & Resources (150+ words) - Key personnel, expertise, and resources committed
    7. Risks & Mitigations (150+ words) - Potential challenges and planned solutions
    8. Evaluation Criteria (150+ words) - How success will be measured

    The proposal MUST be at least 1600 words in total (approximately 3+ pages).
    Include specific details, examples, and quantifiable metrics where possible.
    Use professional language and formal business writing style throughout.
    """,
    ProposalFormat.EXECUTIVE: """
    Generate an executive summary proposal with these requirements:
    {% for item in answers %}
    - {{item.question}}: {{item.answer}}
    {% endfor %}
    
    Focus on strategic value, ROI, and key business benefits.
    Include fully developed sections for:
    1. Executive Overview (200+ words) - High-impact summary tailored for C-level executives
    2. Strategic Background (150+ words) - Brief but substantive context and rationale
    3. Solution Overview (200+ words) - Clear explanation of the proposed solution
    4. Business Impact Analysis (200+ words) - Detailed ROI and strategic advantages
    5. Financial Summary (150+ words) - Clear cost-benefit analysis with key metrics
    6. Timeline Overview (150+ words) - Critical path and key milestones
    7. Recommendation & Next Steps (150+ words) - Clear action items

    The proposal MUST be at least 1200 words total (minimum 2 pages).
    Use executive-appropriate language focusing on business value rather than technical details.
    Include specific metrics, KPIs, and financial projections wherever possible.
    """,
    ProposalFormat.FORMAL: """
    Generate a formal RFP-style proposal with these requirements:
    {% for item in answers %}
    - {{item.question}}: {{item.answer}}
    {% endfor %}
    
    Structure according to standard formal proposal format with fully developed sections:
    1. Cover Page (include title, date, company information)
    2. Executive Summary (250+ words) - Comprehensive yet concise overview
    3. Company Background (200+ words) - Relevant organizational history and qualifications
    4. Understanding of Requirements (250+ words) - Demonstrate clear grasp of client needs
    5. Proposed Solution (300+ words) - Detailed description of recommended approach
    6. Implementation Approach (250+ words) - Step-by-step methodology
    7. Timeline (200+ words) - Detailed timeline with specific dates and deliverables
    8. Budget & Pricing (200+ words) - Comprehensive breakdown with justifications
    9. Terms & Conditions (150+ words) - Clear legal and business terms
    10. Appendices (as needed) - Supporting documentation and details
    
    The proposal MUST be at least 2000 words total (approximately 4+ pages).
    Use formal business language and structure throughout.
    Include specific details, metrics, and quantifiable outcomes.
    Follow standard business proposal formatting with appropriate headings, subheadings, and professional tone.
    """,
}


async def build_proposal_engine(
    session_id: UUID, format: str = ProposalFormat.DETAILED
) -> Any:
    """
    Build a LlamaIndex query engine for generating proposals

    Args:
        session_id: UUID of the session
        format: Format of the proposal (brief, detailed, executive, formal)

    Returns:
        SubQuestionQueryEngine: Query engine for proposal generation
    """
    # 1. Load session data and convert to nodes
    session_storage = SessionStorage(session_id)
    answers_dict = await session_storage.get_answers()

    # Create formatted Q&A nodes
    nodes = []
    for question, answer in answers_dict.items():
        node_text = f"Q: {question}\nA: {answer}"
        nodes.append(TextNode(text=node_text))

    # 2. Configure prompt template based on format
    template_str = PROPOSAL_TEMPLATES.get(
        format, PROPOSAL_TEMPLATES[ProposalFormat.DETAILED]
    )
    prompt = PromptTemplate(template_str)

    # 3. Build vector index from answer nodes
    index = VectorStoreIndex(nodes)

    # 4. Create response synthesizer
    synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",  # Use tree summarization for better structure
        streaming=False,
        text_qa_template=prompt,
    )

    # 5. Create query engine
    query_engine = index.as_query_engine(
        response_synthesizer=synthesizer,
        similarity_top_k=3,  # Retrieve top 3 most relevant answers
    )

    return query_engine


async def generate_proposal(
    session_id: UUID, format: str = ProposalFormat.DETAILED
) -> Dict[str, Any]:
    """
    Generate a proposal based on session data

    Args:
        session_id: UUID of the session
        format: Format of the proposal

    Returns:
        Dict: Generated proposal and metadata
    """
    # Validate format
    if format not in ProposalFormat.values():
        format = ProposalFormat.DETAILED
        logger.warning(f"Invalid format '{format}', using DETAILED instead")

    # Build query engine
    query_engine = await build_proposal_engine(session_id, format)

    # Generate proposal with minimum length requirements
    query = f"Generate a detailed {format} proposal based on the provided answers. Ensure it's comprehensive and at least a full page in length."
    result = await query_engine.aquery(query)

    # Extract response
    proposal_text = result.response

    # Construct result
    return {
        "proposal": proposal_text,
        "format": format,
        "session_id": str(session_id),
        "metadata": {
            "token_usage": getattr(result, "token_usage", None),
            "source_nodes": [
                n.get_content() for n in getattr(result, "source_nodes", [])
            ],
        },
    }


async def generate_proposal_streaming(
    session_id: UUID, format: str = ProposalFormat.DETAILED
):
    """
    Generate a proposal with streaming response

    Args:
        session_id: UUID of the session
        format: Format of the proposal

    Returns:
        Generator: Streaming response generator
    """
    # Same setup as non-streaming version
    if format not in ProposalFormat.values():
        format = ProposalFormat.DETAILED

    # 1. Load session data as nodes
    session_storage = SessionStorage(session_id)
    answers_dict = await session_storage.get_answers()

    nodes = []
    for question, answer in answers_dict.items():
        node_text = f"Q: {question}\nA: {answer}"
        nodes.append(TextNode(text=node_text))

    # 2. Configure streaming response synthesizer
    template_str = PROPOSAL_TEMPLATES.get(
        format, PROPOSAL_TEMPLATES[ProposalFormat.DETAILED]
    )
    prompt = PromptTemplate(template_str)

    # 3. Build index with streaming enabled
    index = VectorStoreIndex(nodes)
    synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        streaming=True,  # Enable streaming
        text_qa_template=prompt,
    )

    # 4. Create streaming query engine
    query_engine = index.as_query_engine(
        response_synthesizer=synthesizer, similarity_top_k=3
    )

    # 5. Generate streaming response with length requirements
    query = f"Generate a comprehensive {format} proposal based on the provided answers. Ensure it's at least a full page in length with detailed sections."
    streaming_response = await query_engine.aquery(query)

    # Return streaming generator
    return streaming_response.response_gen
