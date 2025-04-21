import requests
import json
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

# Update the microservice URL to point to our agent system API
AGENT_API_URL = os.getenv("AGENT_API_URL")  # This is the FastAPI service from our agent system

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
            
            # Log the request for debugging
            print(f"Processing chatbot request: {message[:50]}...")
            print(f"Visualization explicitly requested: {visualization_requested}")
            
            # Call the agent system API
            response = requests.post(
                f'{AGENT_API_URL}/chat/message',
                json={
                    'message': message,
                    'session_id': session_id,
                    'visualization_requested': visualization_requested
                },
                timeout=60  # Increased timeout for complex queries
            )
            
            # Get response data
            response_data = response.json()
            
            # Debug the response data
            print(f"Response keys: {list(response_data.keys())}")
            if 'visualization' in response_data:
                if response_data['visualization'] is not None:
                    viz_data = response_data['visualization']
                    print(f"Visualization data keys: {list(viz_data.keys())}")
                    if 'image_data' in viz_data:
                        print(f"Image data length: {len(viz_data['image_data'])}")
                    else:
                        print("No image_data in visualization")
                else:
                    print("Visualization is None")
            else:
                print("No visualization key in response")
            
            # Prepare response
            result = {
                'message': response_data.get('message', ''),
                'session_id': session_id
            }
            
            # Handle visualization if present
            if response_data.get('visualization'):                     # original scheme
                viz = response_data['visualization']
                result['image_data'] = viz.get('image_data')
                result['image_type'] = viz.get('image_type', 'image/png')

            elif response_data.get('image_data'):                      # flattened scheme from /chat/message
                result['image_data'] = response_data['image_data']
                result['image_type'] = response_data.get('image_type', 'image/png')
            
            return JsonResponse(result)
        except Exception as e:
            print(f"Error in chatbot_message: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)