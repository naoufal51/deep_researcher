"""
Enhanced evaluation script for the research and writing workflow using agentevals.

This script loads test cases from a configuration file and runs comprehensive
evaluations on both components and the complete workflow.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Correct imports for agentevals
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread

# Other necessary imports
from langchain_openai import ChatOpenAI
from langsmith import Client
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

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
    initialize_state_if_needed
)

# Initialize console for pretty output
console = Console()

# Load configuration
def load_config(config_path: str = "eval_config.json") -> Dict:
    """Load test configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        console.print(f"[bold red]Error loading configuration: {e}[/bold red]")
        return {"test_cases": [], "metrics": {}}

# Initialize evaluation model
def init_eval_model(model_name: str = "gpt-4o"):
    """Initialize the LLM for evaluation."""
    return f"openai:{model_name}"

# Create evaluators from config
def create_evaluators(config: Dict, model_name: str):
    """Create evaluators based on configuration."""
    # Extract evaluation prompts from config
    metric_prompts = {}
    for metric_name, metric_data in config.get("metrics", {}).items():
        metric_prompts[metric_name] = metric_data["prompt"]
    
    # Create custom prompt for trajectory evaluation 
    trajectory_prompt = """
    You are an expert evaluator assessing the quality of an AI agent's conversation trajectory.
    
    Analyze the following conversation trajectory:
    
    {outputs}
    
    Consider the following aspects in your evaluation:
    1. Did the agent follow a logical sequence of steps?
    2. Did the agent gather necessary information before providing answers?
    3. Was the final response accurate and complete?
    4. Did the agent use appropriate tools when needed?
    
    Rate the trajectory as either successful (true) or unsuccessful (false).
    Explain your reasoning in detail and provide your final score as true or false.
    """
    
    # Create trajectory evaluator using the factory function
    trajectory_evaluator = create_trajectory_llm_as_judge(
        prompt=trajectory_prompt,
        model=model_name
    )
    
    # Create graph trajectory evaluator
    graph_evaluator = create_graph_trajectory_llm_as_judge(
        model=model_name
    )
    
    return trajectory_evaluator, graph_evaluator

async def test_individual_components(config: Dict, model_name: str):
    """Test each node function individually."""
    console.print("\n[bold blue]=== TESTING INDIVIDUAL COMPONENTS ===[/bold blue]")
    
    # Create a table for results
    table = Table(title="Component Test Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Time (s)", style="yellow")
    table.add_column("Result Snippet", style="white")
    
    # Select the first test case for component testing
    test_case = config["test_cases"][0]
    test_input = {"messages": [{"role": "user", "content": test_case["input"]}]}
    
    components = [
        ("Search Query Generation", generate_search_queries),
        ("Research Agent", run_research_agent),
        ("Math Agent", run_math_agent),
        ("Writing Agent", run_writing_agent),
        ("Publishing Agent", run_publishing_agent),
    ]
    
    # Test each component
    for component_name, component_func in components:
        start_time = time.time()
        try:
            if component_name == "Search Query Generation":
                state = await component_func(test_input)
                result_snippet = f"{len(state['research_state']['search_queries'])} queries generated"
            elif component_name == "Research Agent":
                state = await component_func(test_input)
                # Handle both string and Message objects
                research_results = state["research_state"]["research_results"]
                if hasattr(research_results, 'content'):  # It's a Message object
                    research_results = research_results.content
                result_snippet = research_results[:100] + "..."
            elif component_name == "Math Agent":
                # Math agent needs research results
                research_state = await run_research_agent(test_input)
                state = await component_func(research_state)
                # Handle both string and Message objects
                research_results = state["research_state"]["research_results"]
                if hasattr(research_results, 'content'):  # It's a Message object
                    research_results = research_results.content
                result_snippet = research_results[-150:] + "..."
            elif component_name == "Writing Agent":
                # Writing agent needs research results
                research_state = await run_research_agent(test_input)
                state = await component_func(research_state)
                # Handle both string and Message objects
                final_content = state["writing_state"]["final_content"]
                if hasattr(final_content, 'content'):  # It's a Message object
                    final_content = final_content.content
                result_snippet = final_content[:100] + "..."
            elif component_name == "Publishing Agent":
                # Publishing agent needs both research and writing
                research_state = await run_research_agent(test_input)
                writing_state = await run_writing_agent(research_state)
                state = await component_func(writing_state)
                # Handle both string and Message objects
                final_content = state["writing_state"]["final_content"]
                if hasattr(final_content, 'content'):  # It's a Message object
                    final_content = final_content.content
                result_snippet = final_content[:100] + "..."
            
            duration = time.time() - start_time
            table.add_row(component_name, "✅ Success", f"{duration:.2f}", result_snippet)
        except Exception as e:
            duration = time.time() - start_time
            table.add_row(component_name, f"❌ Error: {str(e)[:50]}", f"{duration:.2f}", "N/A")
    
    console.print(table)

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

async def run_test_case(test_case: Dict, trajectory_evaluator, graph_evaluator):
    """Run a single test case and evaluate results."""
    case_name = test_case["name"]
    input_text = test_case["input"]
    metrics = test_case.get("metrics", [])
    expected_agents = test_case.get("expected_agents", [])
    
    console.print(f"\n[bold cyan]Running test case: {case_name}[/bold cyan]")
    console.print(f"[dim]Input: {input_text}[/dim]")
    
    input_state = {"messages": [{"role": "user", "content": input_text}]}
    thread_id = f"test_{case_name}_{int(time.time())}"
    
    # Run the graph with timing
    start_time = time.time()
    try:
        # Setup LangSmith run if available
        run_manager = None
        try:
            if os.environ.get("LANGCHAIN_API_KEY"):
                client = Client()
                run_manager = client.create_run(
                    name=f"Evaluation: {case_name}",
                    run_type="chain"
                )
        except Exception:
            pass
        
        # Run the graph
        result = await graph.ainvoke(
            input_state, 
            {"configurable": {"thread_id": thread_id}, "run_id": run_manager.id if run_manager else None}
        )
        duration = time.time() - start_time
        
        # Extract final content
        if not result.get("messages"):
            console.print("[bold red]No messages in result![/bold red]")
            return None
        
        # Get the last message content, handling both string and Message objects
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):  # It's a Message object
            final_content = last_message.content
        else:
            final_content = last_message["content"]
        
        console.print(f"[green]Completed in {duration:.2f} seconds[/green]")
        
        # Show content snippet
        content_preview = final_content[:200] + "..." if len(final_content) > 200 else final_content
        console.print("[bold]Output preview:[/bold]")
        console.print(f"[dim]{content_preview}[/dim]")
        
        # Extract trajectory for evaluation
        trajectory = await extract_agent_trajectory(result)
        
        # For graph trajectory, we need to extract from the thread
        try:
            # Check if the function is callable and handle potential issues with await
            if callable(extract_langgraph_trajectory_from_thread):
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    graph_trajectory = await extract_langgraph_trajectory_from_thread(graph, config)
                except TypeError:
                    # If it can't be awaited, try calling it without await
                    graph_trajectory = extract_langgraph_trajectory_from_thread(graph, config)
            else:
                console.print("[yellow]Warning: extract_langgraph_trajectory_from_thread is not callable[/yellow]")
                graph_trajectory = None
        except Exception as e:
            console.print(f"[yellow]Warning: Could not extract graph trajectory: {e}[/yellow]")
            graph_trajectory = None
        
        # Evaluate trajectory using the agentevals trajectory evaluator
        trajectory_eval_results = {}
        try:
            console.print("[cyan]Evaluating agent trajectory...[/cyan]")
            
            # Use the trajectory evaluator directly without metrics - it has its own setup
            trajectory_result = trajectory_evaluator(
                outputs=trajectory
            )
            
            if isinstance(trajectory_result, dict):
                # Format: {'key': 'trajectory_accuracy', 'score': True, 'comment': '...'}
                score = 10 if trajectory_result.get('score') is True else 0
                trajectory_eval_results["trajectory_accuracy"] = {
                    "score": score,
                    "explanation": trajectory_result.get('comment', '')
                }
            
            console.print(f"[green]Trajectory evaluation score: {score}/10[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Trajectory evaluation failed: {e}[/yellow]")
        
        # Evaluate graph trajectory if available
        graph_eval_results = {}
        if graph_trajectory:
            try:
                console.print("[cyan]Evaluating graph trajectory...[/cyan]")
                graph_result = graph_evaluator(
                    inputs=graph_trajectory.get("inputs", []),
                    outputs=graph_trajectory.get("outputs", {})
                )
                
                if isinstance(graph_result, dict):
                    # Format: {'key': 'graph_trajectory_accuracy', 'score': True, 'comment': '...'}
                    graph_score = 10 if graph_result.get('score') is True else 0
                    graph_eval_results["graph_trajectory_accuracy"] = {
                        "score": graph_score,
                        "explanation": graph_result.get('comment', '')
                    }
                
                console.print(f"[green]Graph trajectory evaluation score: {graph_score}/10[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Graph trajectory evaluation failed: {e}[/yellow]")
        
        # Combine all evaluation results
        all_eval_results = {**trajectory_eval_results, **graph_eval_results}
        
        # Compile test results
        return {
            "name": case_name,
            "input": input_text,
            "output": final_content,
            "trajectory": trajectory,
            "duration": duration,
            "evaluations": all_eval_results,
        }
    
    except Exception as e:
        console.print(f"[bold red]Error running test case: {str(e)}[/bold red]")
        return None

async def test_end_to_end_workflow(config: Dict, trajectory_evaluator, graph_evaluator):
    """Test the full workflow with all test cases from config."""
    console.print("\n[bold blue]=== TESTING END-TO-END WORKFLOW ===[/bold blue]")
    
    results = []
    for test_case in config["test_cases"]:
        result = await run_test_case(test_case, trajectory_evaluator, graph_evaluator)
        if result:
            results.append(result)
    
    return results

def print_evaluation_summary(results: List[Dict]):
    """Print a summary table of evaluation results."""
    if not results:
        console.print("[bold red]No evaluation results to display![/bold red]")
        return
    
    console.print("\n[bold blue]=== EVALUATION SUMMARY ===[/bold blue]")
    
    # Create summary table
    table = Table(title="Test Case Evaluation Results")
    table.add_column("Test Case", style="cyan")
    table.add_column("Duration", style="yellow")
    
    # Collect all metrics for columns
    all_metrics = set()
    for result in results:
        all_metrics.update(result["evaluations"].keys())
    
    # Add metric columns
    for metric in sorted(all_metrics):
        table.add_column(metric.capitalize(), style="green")
    
    # Add overall column
    table.add_column("Overall", style="magenta")
    
    # Add rows for each test case
    for result in results:
        row = [result["name"], f"{result['duration']:.2f}s"]
        
        # Calculate scores for each metric
        scores = []
        for metric in sorted(all_metrics):
            if metric in result["evaluations"]:
                score = result["evaluations"][metric]["score"]
                scores.append(score)
                row.append(f"{score}/10")
            else:
                row.append("N/A")
        
        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0
        row.append(f"{overall_score:.1f}/10")
        
        table.add_row(*row)
    
    console.print(table)
    
    # Calculate system-wide average
    all_scores = []
    for result in results:
        scores = [eval_result["score"] for eval_result in result["evaluations"].values()]
        if scores:
            all_scores.extend(scores)
    
    if all_scores:
        system_avg = sum(all_scores) / len(all_scores)
        console.print(f"\n[bold]Overall System Performance: [green]{system_avg:.1f}/10[/green][/bold]")

async def save_results(results: List[Dict], filename_prefix: str = "eval_results"):
    """Save evaluation results to a file."""
    if not results:
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("eval_results", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results/{filename_prefix}_{timestamp}.json"
    
    # Format results for saving
    serializable_results = []
    for result in results:
        serializable_result = {
            "name": result["name"],
            "input": result["input"],
            "output_snippet": result["output"][:500] + "..." if len(result["output"]) > 500 else result["output"],
            "duration": result["duration"],
            "evaluations": {}
        }
        
        for metric, eval_result in result["evaluations"].items():
            serializable_result["evaluations"][metric] = {
                "score": eval_result["score"],
                "explanation": eval_result["explanation"]
            }
        
        serializable_results.append(serializable_result)
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    console.print(f"[green]Results saved to {filename}[/green]")

async def main():
    """Run the full evaluation suite."""
    console.print("[bold]LangGraph Research & Writing Agent Evaluation[/bold]")
    
    # Load configuration
    config = load_config()
    if not config["test_cases"]:
        console.print("[bold red]No test cases found in configuration![/bold red]")
        return
    
    # Initialize evaluation model
    model_name = init_eval_model("gpt-4o")
    
    # Create evaluators
    trajectory_evaluator, graph_evaluator = create_evaluators(config, model_name)
    
    # Test individual components
    await test_individual_components(config, model_name)
    
    # Test end-to-end workflow
    results = await test_end_to_end_workflow(config, trajectory_evaluator, graph_evaluator)
    
    # Print evaluation summary
    print_evaluation_summary(results)
    
    # Save results
    await save_results(results)

if __name__ == "__main__":
    asyncio.run(main()) 