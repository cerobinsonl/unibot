# Testing the University Agent System

This guide explains how to test the multi-agent system before integrating it with your Django frontend.

## Prerequisites

- Docker and Docker Compose installed
- Basic familiarity with terminal/command line
- Postman, cURL, or another API testing tool (optional but recommended)

## Step 1: Start the Services

First, make sure all the services are running:

```bash
cd university-agent-system
docker-compose up -d
```

Check that the services started successfully:

```bash
docker-compose ps
```

You should see the following containers running:
- postgres
- university-api
- university-agent-system

## Step 2: Initialize the Database

Run the database initialization script to create the tables and sample data:

```bash
# Copy the initialization script to the PostgreSQL container
docker cp init-db.sql university-db:/tmp/init-db.sql

# Execute the script
docker exec -it university-db psql -U admin -d university -f /tmp/init-db.sql
```

## Step 3: Verify Health Endpoints

Check if the services are responding correctly:

```bash
# Check FastAPI service
curl http://localhost:8080/health

# Check Agent System
curl http://localhost:8000/health
```

Both should return a JSON response indicating they are healthy.

## Step 4: Test Basic Chat Functionality

Use cURL to test a simple chat request:

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you tell me about your capabilities?", "session_id": "test-session"}'
```

You should receive a response explaining what the system can do.

## Step 5: Test Data Analysis Capabilities

### Basic Data Query

Try a simple data query:

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "How many students are enrolled in each program?", "session_id": "test-session"}'
```

### Data Visualization

Test a visualization request:

```bash
curl -X POST "http://localhost:8080/visualizations/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me a chart of financial aid distribution by aid type", "session_id": "test-session"}'
```

This should return a JSON response containing base64-encoded image data.

To view the raw image directly:

```bash
curl -X POST "http://localhost:8080/visualizations/raw/png" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me a chart of financial aid distribution by aid type", "session_id": "test-session"}' \
  --output chart.png
```

This will save the chart as `chart.png` in your current directory.

## Step 6: Test Data Management Capabilities

### Insert Data

Test data insertion:

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Add a new person with FirstName=John, LastName=Doe, EmailAddress=john.doe@example.com", "session_id": "test-session"}'
```

### Query Inserted Data

Verify the data was inserted:

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Find the person with email john.doe@example.com", "session_id": "test-session"}'
```

## Step 7: Test External System Integration

Test retrieving data from external systems:

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Get the financial aid information for the current academic year", "session_id": "test-session"}'
```

```bash
curl -X POST "http://localhost:8080/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show current enrollment statistics from the SIS", "session_id": "test-session"}'
```

## Step 8: Test WebSocket Connection (Advanced)

For testing WebSockets, you can use a tool like `websocat`:

```bash
# Install websocat if needed
# For Ubuntu: sudo apt-get install websocat
# For macOS: brew install websocat

# Connect to WebSocket
echo '{"message": "Show me the latest enrollment data", "session_id": "ws-test"}' | websocat ws://localhost:8080/ws/chat/ws-test
```

Alternatively, you can use the browser's developer console to test WebSockets:

```javascript
// Run this in your browser's console
const socket = new WebSocket('ws://localhost:8080/ws/chat/browser-test');
socket.onopen = function(e) {
  console.log("Connection established");
  socket.send(JSON.stringify({
    message: "Show me the latest enrollment data",
    session_id: "browser-test"
  }));
};
socket.onmessage = function(event) {
  console.log("Received data:", event.data);
};
```

## Step 9: Interactive Testing with Postman

For more interactive testing, you can use Postman:

1. Create a new request collection for your University Agent System
2. Add a POST request to `http://localhost:8080/chat/message`
3. Set the Content-Type header to `application/json`
4. Add a request body:
   ```json
   {
     "message": "Your test message here",
     "session_id": "postman-test"
   }
   ```
5. Save and send the request

Postman allows you to save different requests and examine responses in detail, including headers and response times.

## Step 10: Test with Sample Conversation Flows

Here are some sample conversation flows to test the agent's capabilities:

### Academic Information Flow

```
Query 1: "How many students are enrolled in Computer Science?"
Query 2: "What is their average GPA?"
Query 3: "Show me a chart of GPA distribution"
```

### Financial Aid Flow

```
Query 1: "What types of financial aid are available?"
Query 2: "Show me the distribution of financial aid awards"
Query 3: "How many students received scholarships last year?"
```

### Student Registration Flow

```
Query 1: "What courses are available for the Fall semester?"
Query 2: "How many seats are left in CS101?"
Query 3: "Register student John Doe for CS101"
```

## Troubleshooting

### View Logs

If you encounter issues, check the container logs:

```bash
# View API logs
docker-compose logs api

# View Agent System logs
docker-compose logs agent-system

# View database logs
docker-compose logs postgres
```

### Restart Services

If a service isn't responding correctly, try restarting it:

```bash
docker-compose restart api
docker-compose restart agent-system
```

### Check Database Connection

Verify the database connection:

```bash
docker exec -it university-db psql -U admin -d university -c "SELECT NOW();"
```

If this command returns the current timestamp, the database connection is working.

## Next Steps: Frontend Integration

Once you've verified that the system is working correctly through API testing, you can proceed with integrating it into your Django frontend as outlined in the QUICKSTART.md guide.

The main integration points will be:

1. Making API calls to `http://localhost:8080/chat/message` for text interactions
2. Setting up visualization display for the base64-encoded image data
3. Implementing WebSocket connections for real-time updates (optional)