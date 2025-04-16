# University Administrative Chatbot System

A hierarchical multi-agent system for university administration that allows non-technical staff to interact with databases, analyze data, and perform administrative tasks using natural language.

## Architecture

The system is built with a hierarchical agent architecture using LangGraph:

- **Director Agent**: Interprets user intent and coordinates responses
- **Coordinator Agents**: Manage specialized domains (Data Analysis, Communication, Data Management, Integration)
- **Specialist Agents**: Perform specific tasks (SQL queries, data analysis, visualization, etc.)

## Features

- Natural language interface for database queries and analysis
- Data visualization generation
- FastAPI-based API with WebSocket support
- Integration with existing Django frontend
- Docker containerization for easy deployment

## Prerequisites

- Docker and Docker Compose
- PostgreSQL database (can be run in a container)
- OpenAI API key

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/university-agent-system.git
cd university-agent-system
```

2. Create a `.env` file with your configuration:
```bash
# Database Configuration
DB_USER=admin
DB_PASSWORD=your_secure_password
DB_NAME=university

# LLM API Keys
OPENAI_API_KEY=your-openai-api-key-here

# Service Configuration
API_DEBUG=true
AGENT_DEBUG=true
```

3. Build and start the services:
```bash
docker-compose up -d
```

4. Initialize the database (if needed):
```bash
docker-compose exec postgres psql -U admin -d university -f /app/db_init/schema.sql
```

## Usage

### Connecting from Django

To integrate with your existing Django frontend, add the following configuration to your Django settings:

```python
# settings.py
CHATBOT_API_URL = "http://localhost:8080"
CHATBOT_WS_URL = "ws://localhost:8080/ws"
```

Then use the provided Django client to interact with the chatbot:

```python
from chatbot_client import ChatbotClient

client = ChatbotClient()
response = client.send_message("Show me enrollment statistics by department")
```

### Direct API Access

You can interact with the system directly via the FastAPI endpoints:

- **Chat Message**: `POST /chat/message`
- **WebSocket**: `ws://localhost:8080/ws/chat/{session_id}`

Example curl request:
```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me the distribution of grades in Biology courses", "session_id": "new-session"}'
```

## Development

### Project Structure

```
university-agent-system/
├── api/                  # FastAPI service
├── agent_system/         # LangGraph agent system
├── docker-compose.yml    # Docker composition
├── .env                  # Environment variables
└── README.md             # This file
```

### Adding New Capabilities

To add new specialist agents:

1. Create a new agent file in `agent_system/agents/specialists/`
2. Add the agent to the appropriate coordinator in `agent_system/agents/coordinators/`
3. Update the workflow graph in `agent_system/graph/workflow.py`

## Troubleshooting

### Common Issues

- **Database Connection Errors**: Check that the PostgreSQL container is running and accessible
- **Agent System Errors**: Check the logs with `docker-compose logs agent-system`
- **API Errors**: Check the logs with `docker-compose logs api`

### Debugging

Enable debug mode in `.env` to get more detailed logs:

```
API_DEBUG=true
AGENT_DEBUG=true
```

## License

[MIT License](LICENSE)