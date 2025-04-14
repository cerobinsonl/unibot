# University Administrative Chatbot - Quick Start Guide

This guide will help you get the University Administrative Chatbot system up and running quickly.

## Prerequisites

- Docker and Docker Compose
- Git
- OpenAI API key

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/university-agent-system.git
cd university-agent-system
```

## Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

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

## Step 3: Start the Services

Build and start all services using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database
- FastAPI service
- Agent System service

## Step 4: Initialize the Database (First time only)

For the POC, we'll use a simple database schema. Run the initialization script:

```bash
docker-compose exec postgres psql -U admin -d university -f /app/init-db.sql
```

## Step 5: Verify the Setup

Check if all services are running:

```bash
docker-compose ps
```

Verify the API is accessible:

```bash
curl http://localhost:8080/health
```

You should see: `{"status":"healthy"}`

## Step 6: Try the Chatbot

Send a test message to the chatbot:

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "How many students are enrolled in each department?", "session_id": "test-session"}'
```

## Step 7: Integrate with Django

Add the following to your Django views:

```python
# views.py
import requests
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def chatbot_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            session_id = data.get('session_id', 'default-session')
            
            # Call the FastAPI endpoint
            response = requests.post(
                'http://localhost:8080/chat/message',
                json={
                    'message': message,
                    'session_id': session_id
                }
            )
            
            return JsonResponse(response.json())
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
```

Add URL routing:

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/chatbot/message', views.chatbot_message, name='chatbot_message'),
]
```

## Step 8: Create a Simple Chat Interface

Add a simple chat interface to your Django template:

```html
<!-- chat.html -->
<div class="chat-container">
    <div id="chat-messages"></div>
    
    <div class="chat-input">
        <input type="text" id="user-message" placeholder="Ask a question...">
        <button id="send-btn">Send</button>
    </div>
</div>

<script>
    const sessionId = 'session-' + Math.random().toString(36).substring(2, 10);
    const messagesContainer = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-message');
    const sendButton = document.getElementById('send-btn');
    
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // Add user message to the chat
        addMessage('user', message);
        userInput.value = '';
        
        // Send to backend
        fetch('/api/chatbot/message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            }),
        })
        .then(response => response.json())
        .then(data => {
            // Handle response
            addMessage('assistant', data.message);
            
            // If there's an image, display it
            if (data.image_data) {
                addImage(data.image_data, data.image_type);
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            addMessage('system', 'Error: Could not get response');
        });
    }
    
    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.textContent = content;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function addImage(imageData, imageType) {
        const imgElement = document.createElement('img');
        imgElement.src = `data:${imageType};base64,${imageData}`;
        imgElement.className = 'chat-image';
        messagesContainer.appendChild(imgElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
</script>
```

Add some basic CSS:

```css
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    border: 1px solid #ccc;
    border-radius: 8px;
    overflow: hidden;
}

#chat-messages {
    height: 400px;
    overflow-y: auto;
    padding: 15px;
    background-color: #f9f9f9;
}

.message {
    margin-bottom: 10px;
    padding: 8px 12px;
    border-radius: 5px;
    max-width: 70%;
}

.user {
    background-color: #e3f2fd;
    margin-left: auto;
}

.assistant {
    background-color: #f1f1f1;
}

.system {
    background-color: #ffebee;
    max-width: 100%;
    text-align: center;
}

.chat-image {
    max-width: 100%;
    height: auto;
    border-radius: 5px;
    margin: 10px 0;
}

.chat-input {
    display: flex;
    padding: 10px;
    background-color: #fff;
    border-top: 1px solid #ccc;
}

#user-message {
    flex: 1;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

#send-btn {
    margin-left: 10px;
    padding: 8px 16px;
    background-color: #2196f3;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
```

## Step 9: Test the WebSocket Connection (Optional)

For real-time interactions, you can use WebSockets:

```javascript
// In your Django template
const socket = new WebSocket(`ws://localhost:8080/ws/chat/${sessionId}`);

socket.onopen = (event) => {
    console.log("WebSocket connection established");
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("Message from server:", data);
    
    if (data.type === "result") {
        addMessage('assistant', data.message);
        
        if (data.image_data) {
            addImage(data.image_data, data.image_type);
        }
    }
};

// Send a message via WebSocket
function sendWebSocketMessage(message) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            message: message,
            session_id: sessionId
        }));
    } else {
        console.error("WebSocket is not open");
    }
}
```

## Common Issues and Troubleshooting

### Services Not Starting

Check Docker logs:

```bash
docker-compose logs api
docker-compose logs agent-system
docker-compose logs postgres
```

### Database Connection Issues

Verify database is running:

```bash
docker-compose exec postgres pg_isready -U admin -d university
```

### API Not Responding

Check API logs and restart if needed:

```bash
docker-compose logs api
docker-compose restart api
```

### LLM API Errors

Verify your OpenAI API key in the `.env` file.

## Next Steps

- Explore the full API documentation at `http://localhost:8080/docs`
- Add more specialized agents for your university's needs
- Customize the chat interface for better user experience
- Implement user authentication and role-based access