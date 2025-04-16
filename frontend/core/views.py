import requests
import json
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Update the microservice URL to point to our agent system API
AGENT_API_URL = 'http://api:8080'  # This is the FastAPI service from our agent system

def home(request):
    """
    Home page with cards and chatbot interface
    """
    return render(request, 'core/home.html')

def data_analysis_view(request):
    """
    Data Analysis page with form to analyze university data
    """
    return render(request, 'core/data_analysis.html')

def send_messages_view(request):
    """
    Send Messages page with form to send communications
    """
    return render(request, 'core/send_messages.html')

def input_data_view(request):
    """
    Input Data page with form to add data to the database
    """
    return render(request, 'core/input_data.html')

def extract_data_view(request):
    """
    Extract Data page with options to pull data from external systems
    """
    return render(request, 'core/extract_data.html')

def create_synthetic_data_view(request):
    """
    Create Synthetic Data page for generating test data
    """
    return render(request, 'core/create_synthetic_data.html')

@csrf_exempt
def chatbot_message(request):
    """
    API endpoint to handle chatbot messages
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            session_id = data.get('session_id', f'session-{uuid.uuid4()}')
            
            # Determine if visualization is explicitly requested
            visualization_requested = any(keyword in message.lower() for keyword in 
                ['chart', 'plot', 'graph', 'visualization', 'visualize', 'visualisation', 
                 'histogram', 'bar chart', 'show me', 'display'])
            
            # Call the agent system API
            response = requests.post(
                f'{AGENT_API_URL}/chat/message',
                json={
                    'message': message,
                    'session_id': session_id,
                    'visualization_requested': visualization_requested
                },
                timeout=30  # Increased timeout for complex queries
            )
            
            # Get response data
            response_data = response.json()
            
            # If message explicitly requests visualization but none was provided
            if visualization_requested and ('has_visualization' not in response_data or not response_data.get('has_visualization')):
                try:
                    # Try to get a visualization specifically
                    viz_response = requests.post(
                        f'{AGENT_API_URL}/visualizations/generate',
                        json={
                            'query': message,
                            'session_id': session_id
                        },
                        timeout=30
                    )
                    
                    if viz_response.status_code == 200:
                        viz_data = viz_response.json()
                        if 'image_data' in viz_data:
                            response_data['image_data'] = viz_data['image_data']
                            response_data['image_type'] = viz_data.get('image_type', 'image/png')
                except Exception as viz_error:
                    # If visualization fails, just log the error and continue
                    print(f"Visualization error: {viz_error}")
            
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)