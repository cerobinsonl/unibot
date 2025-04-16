#!/usr/bin/env python3
import argparse
import requests
import json
import base64
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Debug visualization transfer')
    parser.add_argument('--message', '-m', default='Show me a visualization of student enrollment by program', 
                       help='Message to send')
    parser.add_argument('--session', '-s', default=f'debug-{datetime.now().strftime("%Y%m%d%H%M%S")}', 
                       help='Session ID')
    parser.add_argument('--agent-url', default='http://localhost:8000/process', 
                       help='Direct agent system URL')
    parser.add_argument('--api-url', default='http://localhost:8080/chat/message', 
                       help='API URL')
    parser.add_argument('--save-image', '-i', action='store_true', 
                       help='Save any received images')
    return parser.parse_args()

def call_agent_directly(url, message, session_id):
    print(f"Calling agent system directly at {url}")
    
    payload = {
        "message": message,
        "session_id": session_id
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling agent: {e}")
        return None

def call_api(url, message, session_id):
    print(f"Calling API at {url}")
    
    payload = {
        "message": message,
        "session_id": session_id
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

def save_image(image_data, image_type, prefix):
    if not os.path.exists('debug_images'):
        os.makedirs('debug_images')
    
    file_ext = image_type.split('/')[-1] if '/' in image_type else 'png'
    filename = f"debug_images/{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_ext}"
    
    try:
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(image_data))
        print(f"Image saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def analyze_response(response, source, save_imgs=False):
    print(f"\n===== {source.upper()} RESPONSE ANALYSIS =====")
    
    if not response:
        print("No response received")
        return
    
    print(f"Response keys: {list(response.keys())}")
    
    # Check for regular keys
    for key in ['message', 'session_id']:
        if key in response:
            value = response[key]
            if isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")
    
    # Check for visualization
    visualization_found = False
    
    # Option 1: Direct visualization object
    if 'visualization' in response:
        visualization_found = True
        viz = response['visualization']
        print("Visualization object found directly in response")
        
        if isinstance(viz, dict):
            print(f"Visualization keys: {list(viz.keys())}")
            
            if 'image_data' in viz and viz['image_data']:
                print(f"image_data present, length: {len(viz['image_data'])}")
                if save_imgs:
                    save_image(viz['image_data'], viz.get('image_type', 'image/png'), f"{source}_direct")
            else:
                print("image_data missing or empty")
    
    # Option 2: image_data at root level
    elif 'image_data' in response:
        visualization_found = True
        print("image_data found at root level")
        print(f"image_data length: {len(response['image_data'])}")
        
        if save_imgs:
            save_image(response['image_data'], response.get('image_type', 'image/png'), f"{source}_root")
    
    # Option 3: has_visualization flag
    elif 'has_visualization' in response:
        if response['has_visualization']:
            visualization_found = True
            print("has_visualization flag is True, but no image data found")
        else:
            print("has_visualization flag is False")
    
    if not visualization_found:
        print("No visualization data found in response")
    
    return visualization_found

def main():
    args = parse_args()
    
    # Step 1: Call agent system directly
    agent_response = call_agent_directly(args.agent_url, args.message, args.session)
    agent_has_viz = analyze_response(agent_response, "agent", args.save_image)
    
    # Step 2: Call through API
    api_response = call_api(args.api_url, args.message, args.session)
    api_has_viz = analyze_response(api_response, "api", args.save_image)
    
    # Summary
    print("\n===== VISUALIZATION TRANSFER SUMMARY =====")
    print(f"Agent system has visualization: {agent_has_viz}")
    print(f"API response has visualization: {api_has_viz}")
    
    if agent_has_viz and not api_has_viz:
        print("\nPROBLEM DETECTED: Visualization is present in agent response but missing in API response")
        print("This suggests the API is not correctly passing the visualization data from the agent system.")
        print("\nPossible solutions:")
        print("1. Check the API router code to ensure it properly extracts visualization data from agent response")
        print("2. Ensure visualization is directly included in the API response")
        print("3. Verify the agent system is returning the visualization in the expected format")
    elif not agent_has_viz:
        print("\nPROBLEM DETECTED: Visualization is missing in agent system response")
        print("This suggests the agent system is not creating a visualization or not including it in the response.")
        print("\nPossible solutions:")
        print("1. Check the visualization agent to ensure it's creating visualizations properly")
        print("2. Verify that visualizations are being preserved in the workflow state")
        print("3. Confirm the agent system is including visualization data in its response")
    elif agent_has_viz and api_has_viz:
        print("\nBoth agent system and API have visualization data")
        print("If visualizations still aren't appearing, check the client-side code that displays them")

if __name__ == "__main__":
    main()