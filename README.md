# LlamaIndex Proposal Generator Microservice

A powerful microservice for generating professional proposals using LlamaIndex, optimized for dynamic Q&A-based proposal generation.

## Features

- Dynamic Q&A flow for proposal generation
- Structured data validation and business rules enforcement
- Multi-LLM support with intelligent fallback
- Real-time proposal streaming
- Versioned proposal history
- Enterprise-grade security and monitoring

## Architecture

The service follows a microservice architecture with the following components:

- **API Layer**: FastAPI-based REST endpoints
- **Query Planner**: Structures Q&A flow using LlamaIndex
- **Response Synthesizer**: Generates LLM responses with context
- **Template Engine**: Maps answers to proposal sections
- **Validation Module**: Enforces answer constraints
- **LLM Gateway**: Multi-LLM abstraction layer
- **History Store**: Versioned proposal drafts

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis
- OpenAI API key or other LLM provider credentials

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd proposal-generator
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
or 

```bash
pip install -e .
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Create a `.env` file with the following variables:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-api-key

# Database
POSTGRES_USER=proposals
POSTGRES_PASSWORD=your-password
POSTGRES_DB=proposals
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# LLM Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEFAULT_LLM=gpt-4-turbo
```

## Running the Service

### Using Docker Compose

```bash
docker-compose -f docker-compose.llamaindex.yml up -d
```

### Local Development

```bash
uvicorn proposals.main:app --reload
```

## API Endpoints

### Session Management

- `POST /sessions` - Initialize a new Q&A session
- `POST /answers` - Process and validate answers
- `POST /generate` - Generate proposal from session data
- `GET /proposals/stream` - Stream proposal generation

## Usage Example

```python
import requests

# Initialize session
session = requests.post("http://localhost:8000/sessions",
                       json={"user_id": "user123"})
session_id = session.json()["session_id"]

# Submit answers
answer = requests.post("http://localhost:8000/answers",
                      json={
                          "session_id": session_id,
                          "question": "What is your budget?",
                          "answer": "50000"
                      })

# Generate proposal
proposal = requests.post("http://localhost:8000/generate",
                        json={
                            "session_id": session_id,
                            "format": "detailed"
                        })
```

## Monitoring

The service includes built-in monitoring using Prometheus metrics:

- Query planning accuracy
- LLM usage and costs
- Proposal quality metrics
- Performance metrics (latency, error rates)

## Security

- API key authentication
- Rate limiting per API key
- SOC2-compliant data handling
- Secure credential management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
