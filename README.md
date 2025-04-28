# Proposal Generator with LlamaIndex + Gemini

A microservice for generating customized proposals based on Q&A interactions, powered by LlamaIndex and Google's Gemini LLM.

## 🚀 Features

- **Q&A-Driven Proposal Generation**: Generate proposals from user-provided answers
- **Dynamic Question Generation**: Smart follow-up questions based on previous answers
- **Multiple Proposal Formats**: Brief, detailed, executive, and formal templates
- **Validation Rules**: Business rule enforcement for answer validation
- **Streaming Responses**: Real-time proposal generation with event streaming
- **Gemini LLM Integration**: Leverages Google's Gemini LLM for high-quality content

## 📋 Architecture

The system follows a microservice architecture as detailed in `docs.txt`:

- **API Layer**: FastAPI endpoints for interaction
- **Query Planner**: Structures Q&A flow as retrieval tasks
- **Response Synthesizer**: Generates content with context
- **Template Engine**: Maps answers to proposal sections
- **Validation Module**: Enforces business rules

## 🛠️ Setup

### Prerequisites

- Python 3.9+
- Google API key for Gemini

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/proposal-generator.git
   cd proposal-generator
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix or MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   DEBUG=true
   LOG_LEVEL=INFO
   ```

### Running the Application

Start the FastAPI server:

```bash
python main.py
```

The API will be available at http://localhost:8000.

API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧩 API Usage

### Creating a Session

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "user_name": "John Doe"}'
```

### Answering Questions

```bash
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/answers \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the name of the project?",
    "answer": "Website Redesign Project",
    "question_type": "GENERAL"
  }'
```

### Generating a Proposal

```bash
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/proposals \
  -H "Content-Type: application/json" \
  -d '{
    "format": "detailed",
    "include_metadata": true
  }'
```

## 📚 Documentation Links

- [Response Synthesizer](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/)
- [Query Planner](https://docs.llamaindex.ai/en/stable/examples/workflow/planning_workflow/)
- [Vector Store Index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/)
- [Evaluation](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/evaluation/)
- [Streaming](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming/)
- [Gemini Integration](https://docs.llamaindex.ai/en/stable/examples/llm/gemini/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
