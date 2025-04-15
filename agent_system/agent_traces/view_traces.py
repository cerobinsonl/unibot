#!/usr/bin/env python3
import json
import argparse
import os
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='View agent traces')
    parser.add_argument('trace_file', nargs='?', help='Trace file to view')
    parser.add_argument('--list', '-l', action='store_true', help='List available trace files')
    parser.add_argument('--dir', '-d', default='./agent_traces', help='Directory containing trace files')
    return parser.parse_args()

def list_trace_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return []
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    files.sort(reverse=True)  # Most recent first
    
    print(f"Found {len(files)} trace files:")
    for i, file in enumerate(files):
        try:
            with open(os.path.join(directory, file), 'r') as f:
                trace = json.load(f)
                start_time = trace.get('start_time', 'Unknown')
                visualization = "✓" if trace.get('visualization_created', False) else "✗"
                print(f"{i+1}. {file} - {start_time} [Visualization: {visualization}]")
        except Exception as e:
            print(f"{i+1}. {file} - Error: {e}")
    
    return files

def display_trace(trace_path):
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    
    print("\n===== TRACE SUMMARY =====")
    print(f"Start Time: {trace.get('start_time', 'Unknown')}")
    print(f"End Time: {trace.get('end_time', 'Unknown')}")
    print(f"Visualization Created: {trace.get('visualization_created', False)}")
    print(f"Agents Involved: {', '.join(trace.get('agents', {}).keys())}")
    
    print("\n===== AGENT INTERACTIONS =====")
    for i, msg in enumerate(trace.get('messages', [])):
        print(f"{i+1}. {msg.get('agent')} → {msg.get('action')} ({msg.get('timestamp')})")
    
    print("\n===== STATE UPDATES =====")
    for i, state in enumerate(trace.get('state_updates', [])):
        print(f"{i+1}. Agent: {state.get('current_agent')}")
        print(f"   Has visualization: {state.get('has_visualization')}")
        print(f"   Step count: {state.get('step_count')}")
    
    if 'final_state' in trace:
        print("\n===== FINAL STATE =====")
        final = trace['final_state']
        print(f"Has response: {final.get('has_response')}")
        print(f"Has visualization: {final.get('has_visualization')}")
        
        if 'visualization_details' in final:
            viz = final['visualization_details']
            print(f"Chart type: {viz.get('chart_type')}")
            print(f"Has image data: {viz.get('has_image_data')}")
            print(f"Image type: {viz.get('image_type')}")

def main():
    args = parse_args()
    
    if args.list:
        list_trace_files(args.dir)
        return
    
    if args.trace_file:
        if os.path.exists(args.trace_file):
            display_trace(args.trace_file)
        elif os.path.exists(os.path.join(args.dir, args.trace_file)):
            display_trace(os.path.join(args.dir, args.trace_file))
        else:
            print(f"Trace file not found: {args.trace_file}")
            files = list_trace_files(args.dir)
            if files:
                print("\nEnter the number of the trace you want to view:")
                try:
                    choice = int(input("> ")) - 1
                    if 0 <= choice < len(files):
                        display_trace(os.path.join(args.dir, files[choice]))
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Invalid input")
    else:
        files = list_trace_files(args.dir)
        if files:
            print("\nEnter the number of the trace you want to view:")
            try:
                choice = int(input("> ")) - 1
                if 0 <= choice < len(files):
                    display_trace(os.path.join(args.dir, files[choice]))
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid input")

if __name__ == "__main__":
    main()