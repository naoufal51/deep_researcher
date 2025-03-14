"""
Quick test script for the research and writing workflow using agentevals.

This script runs a single test case to verify that the evaluation works properly.
"""

import asyncio
import os
import json
import traceback
import uuid
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from src.react_agent.graph import graph

from agentevals.trajectory.llm import create_trajectory_llm_as_judge

from src.react_agent.graph import (
    run_research_agent,
    run_writing_agent,
    run_publishing_agent
)

# Initialize evaluation model
MODEL_NAME = "openai:gpt-4o"

# Simple test case
TEST_CASE = {
    "name": "quick_test",
    "input": "What are the benefits of solar energy?",
}

# Create trajectory evaluator
trajectory_evaluator = create_trajectory_llm_as_judge(
    model=MODEL_NAME
)

async def extract_agent_trajectory(state):
    """Extract the agent trajectory from the state."""
    # Convert the state to a message trajectory format that agentevals can use
    messages = state.get("messages", [])
    
    # Convert each message to a dict format that agentevals can understand
    trajectory = []
    for msg in messages:
        if hasattr(msg, 'type'):  # It's a LangChain Message object
            # Skip system messages
            if msg.type == 'system':
                continue
                
            # Convert to dict format
            msg_dict = {
                "role": msg.type if msg.type != "ai" else "assistant",
                "content": msg.content
            }
            
            # Add tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
                
            trajectory.append(msg_dict)
        else:
            # Already in dict format
            if msg.get("role") != "system":
                trajectory.append(msg)
    
    return trajectory

async def run_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case and return the results."""
    try:
        # Create input for the graph
        input_message = HumanMessage(content=test_case["input"])
        
        # Run the graph
        print(f"Running test: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        
        # Generate a unique thread ID for this test
        thread_id = f"test_{test_case['name']}_{uuid.uuid4()}"
        
        # Execute the graph with required configurable parameters
        result = await graph.ainvoke(
            {"messages": [input_message]},
            {"configurable": {"thread_id": thread_id}}
        )
        
        # Debug: Print the result structure
        print("\nResult structure:")
        if isinstance(result, dict):
            for key in result.keys():
                print(f"- {key}: {type(result[key])}")
                
                # If there's feedback, print it
                if key == "feedback" or key == "writing_feedback":
                    feedback = result[key]
                    print(f"\nFeedback for {feedback.get('agent_name', 'unknown')}:")
                    print(f"- Feedback: {feedback.get('feedback', 'None')}")
                    
                    # Print trajectory info
                    trajectory = feedback.get('trajectory', [])
                    print(f"- Trajectory length: {len(trajectory)}")
                    
                    # Print sample of trajectory messages
                    for i, msg in enumerate(trajectory[:3]):
                        if hasattr(msg, 'role'):
                            role = msg.role
                            content_type = type(msg.content)
                            content_snippet = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                        elif isinstance(msg, dict):
                            role = msg.get('role', 'unknown')
                            content_type = type(msg.get('content', ''))
                            content = msg.get('content', '')
                            content_snippet = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                        else:
                            role = "unknown"
                            content_type = type(msg)
                            content_snippet = str(msg)[:100] + "..." if len(str(msg)) > 100 else str(msg)
                            
                        print(f"  Message {i}: role={role}, content_type={content_type}, snippet={content_snippet}")
        
        # Extract the final content
        final_content = ""
        if isinstance(result, dict) and "writing_state" in result:
            final_content = result["writing_state"].get("final_content", "")
        
        print("\nFinal Content:")
        print(final_content[:500] + "..." if len(final_content) > 500 else final_content)
        
        return {
            "test_case": test_case,
            "result": result,
            "final_content": final_content,
            "success": True
        }
    except Exception as e:
        print(f"Error running test: {e}")
        traceback.print_exc()
        return {
            "test_case": test_case,
            "error": str(e),
            "success": False
        }

async def main():
    """Run all test cases."""
    result = await run_test(TEST_CASE)
    print("\nTest completed.")
    if result["success"]:
        print("✅ Test passed")
    else:
        print("❌ Test failed")
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main()) 