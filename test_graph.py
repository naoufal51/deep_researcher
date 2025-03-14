"""
Evaluation script for the research and writing workflow using agentevals.

This script tests both individual components and end-to-end workflow performance,
using LLM-based evaluators to assess the quality of outputs.
"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional

# Correct imports for agentevals
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT 
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread

from langchain_openai import ChatOpenAI
from langsmith import Client

# Import our graph and its components
from src.react_agent.graph import (
    # Main graph
    graph,
    # Core functions
    run_research_agent,
    run_math_agent,
    run_writing_agent,
    run_publishing_agent,
    generate_search_queries,
    # Helper functions
    initialize_state_if_needed,
    add_message_to_state
)

# Initialize evaluation model string using OpenAI format
MODEL_NAME = "openai:gpt-4o"

# Test cases to evaluate the agent
TEST_CASES = [
    {
        "name": "general_research_query",
        "input": {"messages": [{"role": "user", "content": "What are the environmental impacts of electric vehicles compared to traditional combustion engine vehicles?"}]},
        "expected_nodes": ["research_agent", "writing_agent", "publishing_agent"],
        "quality_metrics": ["accuracy", "thoroughness", "coherence"]
    },
    {
        "name": "math_focused_query",
        "input": {"messages": [{"role": "user", "content": "Calculate the carbon footprint savings when switching from a gasoline car to an electric vehicle over 5 years."}]},
        "expected_nodes": ["research_agent", "math_agent", "writing_agent", "publishing_agent"],
        "quality_metrics": ["accuracy", "calculation_correctness", "coherence"]
    },
    {
        "name": "complex_query_requiring_research_and_analysis",
        "input": {"messages": [{"role": "user", "content": "Analyze the economic implications of transitioning to renewable energy sources for developing countries by 2030."}]},
        "expected_nodes": ["research_agent", "math_agent", "writing_agent", "publishing_agent"],
        "quality_metrics": ["accuracy", "thoroughness", "coherence", "calculation_correctness"]
    },
]

# Create the trajectory evaluator using the factory function
trajectory_evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    model=MODEL_NAME,
)

# Create graph trajectory evaluator
graph_trajectory_evaluator = create_graph_trajectory_llm_as_judge(
    model=MODEL_NAME,
)

async def extract_agent_trajectory(state):
    """Extract the agent trajectory from a completed state."""
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

async def test_individual_components():
    """Test each node function individually to isolate issues."""
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===")
    
    # Test case for individual components
    test_input = {"messages": [{"role": "user", "content": "What are the latest developments in renewable energy?"}]}
    thread_id = f"component_test_{int(time.time())}"
    
    # Test search query generation
    print("\nTesting search query generation...")
    try:
        state = await generate_search_queries(test_input)
        queries = state["research_state"]["search_queries"]
        print(f"Generated {len(queries)} search queries:")
        for q in queries:
            print(f"- {q.search_query}")
    except Exception as e:
        print(f"Error in search query generation: {e}")
    
    # Test research agent
    print("\nTesting research agent...")
    try:
        state = await run_research_agent(test_input)
        research_results = state["research_state"]["research_results"]
        print(f"Research results snippet: {research_results[:200]}...")
    except Exception as e:
        print(f"Error in research agent: {e}")
    
    # Test writing agent
    print("\nTesting writing agent...")
    try:
        # Need to prepare state with research results
        research_state = await run_research_agent(test_input)
        writing_state = await run_writing_agent(research_state)
        draft_content = writing_state["writing_state"]["final_content"]
        print(f"Draft content snippet: {draft_content[:200]}...")
    except Exception as e:
        print(f"Error in writing agent: {e}")
    
    # Test publishing agent
    print("\nTesting publishing agent...")
    try:
        # Need complete pipeline up to writing
        research_state = await run_research_agent(test_input)
        writing_state = await run_writing_agent(research_state)
        final_state = await run_publishing_agent(writing_state)
        final_content = final_state["writing_state"]["final_content"]
        print(f"Final content snippet: {final_content[:200]}...")
    except Exception as e:
        print(f"Error in publishing agent: {e}")

async def test_end_to_end_workflow():
    """Test the full workflow with different inputs and capture trajectories."""
    print("\n=== TESTING END-TO-END WORKFLOW ===")
    
    all_results = []
    
    for test_case in TEST_CASES:
        print(f"\nRunning test case: {test_case['name']}")
        input_state = test_case["input"]
        thread_id = f"end_to_end_{test_case['name']}_{int(time.time())}"
        
        try:
            # Run the workflow with tracing enabled (this requires LangSmith API key)
            # If you don't have LangSmith access, the graph will still run but without tracing
            run_manager = None
            try:
                if os.environ.get("LANGCHAIN_API_KEY"):
                    client = Client()
                    run_manager = client.create_run(
                        name=f"Evaluation: {test_case['name']}",
                        run_type="chain"
                    )
            except Exception:
                pass  # Continue without tracing if not available
            
            # Run the graph
            result = await graph.ainvoke(
                input_state, 
                {"configurable": {"thread_id": thread_id}, "run_id": run_manager.id if run_manager else None}
            )
            
            # Extract final content
            final_content = result["messages"][-1]["content"] if result.get("messages") else "No output generated"
            
            print(f"Output snippet: {final_content[:200]}...")
            
            # Extract trajectory for evaluation using agentevals
            trajectory = await extract_agent_trajectory(result)
            
            # Try to extract the graph trajectory for graph-based evaluation
            try:
                if callable(extract_langgraph_trajectory_from_thread):
                    config = {"configurable": {"thread_id": thread_id}}
                    try:
                        graph_trajectory = await extract_langgraph_trajectory_from_thread(graph, config)
                    except TypeError:
                        # If it can't be awaited, try calling it without await
                        graph_trajectory = extract_langgraph_trajectory_from_thread(graph, config)
                else:
                    print("Warning: extract_langgraph_trajectory_from_thread is not callable")
                    graph_trajectory = None
            except Exception as e:
                print(f"Warning: Could not extract graph trajectory: {e}")
                graph_trajectory = None
            
            # Evaluate the trajectory using the trajectory evaluator
            print("Evaluating agent trajectory...")
            trajectory_result = trajectory_evaluator(
                outputs=trajectory
            )
            
            trajectory_score = "Pass" if trajectory_result.get('score') is True else "Fail"
            print(f"Trajectory evaluation: {trajectory_score}")
            print(f"Explanation: {trajectory_result.get('comment', '')[:200]}...")
            
            # Evaluate graph trajectory if available
            graph_trajectory_result = None
            if graph_trajectory:
                print("Evaluating graph trajectory...")
                graph_trajectory_result = graph_trajectory_evaluator(
                    inputs=graph_trajectory.get("inputs", []),
                    outputs=graph_trajectory.get("outputs", {})
                )
                
                graph_score = "Pass" if graph_trajectory_result.get('score') is True else "Fail"
                print(f"Graph trajectory evaluation: {graph_score}")
                print(f"Explanation: {graph_trajectory_result.get('comment', '')[:200]}...")
            
            # Store the evaluation results
            all_results.append({
                "test_case": test_case["name"],
                "input": input_state,
                "output": final_content,
                "trajectory": trajectory,
                "trajectory_evaluation": trajectory_result,
                "graph_trajectory_evaluation": graph_trajectory_result
            })
            
        except Exception as e:
            print(f"Error in end-to-end workflow for {test_case['name']}: {e}")
    
    return all_results

async def main():
    """Run all tests and evaluations."""
    # Set up LangSmith for tracing if available
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("NOTE: For full tracing capabilities, set the LANGCHAIN_API_KEY environment variable")
    
    # Test individual components
    await test_individual_components()
    
    # Test end-to-end workflow
    results = await test_end_to_end_workflow()
    
    # Calculate overall scores
    if results:
        print("\n=== SUMMARY OF EVALUATIONS ===")
        trajectory_scores = []
        graph_scores = []
        
        for result in results:
            test_name = result["test_case"]
            
            # For trajectory eval
            traj_result = result.get("trajectory_evaluation", {})
            traj_score = True if traj_result.get("score") is True else False
            trajectory_scores.append(traj_score)
            
            # For graph trajectory eval
            graph_result = result.get("graph_trajectory_evaluation", {})
            if graph_result:
                graph_score = True if graph_result.get("score") is True else False
                graph_scores.append(graph_score)
            
            print(f"{test_name}:")
            print(f"  - Trajectory evaluation: {'Pass' if traj_score else 'Fail'}")
            if graph_result:
                print(f"  - Graph trajectory evaluation: {'Pass' if graph_score else 'Fail'}")
        
        # Calculate pass percentages
        trajectory_pass_rate = sum(1 for s in trajectory_scores if s) / len(trajectory_scores) * 100 if trajectory_scores else 0
        graph_pass_rate = sum(1 for s in graph_scores if s) / len(graph_scores) * 100 if graph_scores else 0
        
        print(f"\nTrajectory evaluation pass rate: {trajectory_pass_rate:.1f}%")
        if graph_scores:
            print(f"Graph trajectory evaluation pass rate: {graph_pass_rate:.1f}%")

if __name__ == "__main__":
    import time  # Import time module for creating unique thread IDs
    asyncio.run(main()) 