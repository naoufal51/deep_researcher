"""Test script for the LangGraph implementation with procedural memory."""
import asyncio
from src.react_agent.graph import graph, memory_store

async def test_memory_optimization():
    """Test procedural memory and prompt optimization."""
    print("Initial agent prompts:")
    print("-" * 50)
    for agent_name in ["research_agent", "writing_agent", "publishing_agent", "math_agent"]:
        try:
            item = memory_store.get((("instructions",)), key=agent_name)
            prompt = item.value["prompt"]
            print(f"{agent_name} prompt:")
            print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
        except (KeyError, AttributeError) as e:
            print(f"Error accessing {agent_name} prompt: {str(e)}")
        print("-" * 50)
    
    # Test the graph with a sample query
    print("\nRunning test query...")
    result = await graph.ainvoke({
        "messages": [
            {"role": "user", "content": "What are the main environmental impacts of switching from fossil fuels to renewable energy?"}
        ]
    })
    
    # Print the result
    print("\nAgent response:")
    print("-" * 50)
    if "messages" in result and result["messages"]:
        for message in result["messages"]:
            if isinstance(message, dict) and "role" in message and message["role"] == "assistant":
                print(message["content"])
    print("-" * 50)
    
    # Now test feedback processing
    print("\nProcessing sample feedback...")
    feedback_result = await graph.ainvoke({
        "messages": [
            {"role": "user", "content": "Feedback: Please include more specific data points and statistics in your research results."}
        ]
    })
    
    # Print the feedback response
    print("\nFeedback acknowledgment:")
    print("-" * 50)
    if "messages" in feedback_result and feedback_result["messages"]:
        for message in feedback_result["messages"]:
            if isinstance(message, dict) and "role" in message and message["role"] == "assistant":
                print(message["content"])
    print("-" * 50)
    
    # Check if prompts were optimized
    print("\nOptimized agent prompts:")
    print("-" * 50)
    for agent_name in ["research_agent", "writing_agent", "publishing_agent", "math_agent"]:
        try:
            item = memory_store.get((("instructions",)), key=agent_name)
            prompt = item.value["prompt"]
            print(f"{agent_name} prompt:")
            print(prompt[:100] + "..." if len(prompt) > 100 else prompt)
        except (KeyError, AttributeError) as e:
            print(f"Error accessing {agent_name} prompt: {str(e)}")
        print("-" * 50)

    # Test the graph again to see if behavior improved
    print("\nRunning test query after optimization...")
    result = await graph.ainvoke({
        "messages": [
            {"role": "user", "content": "What are the main environmental impacts of solar panel manufacturing?"}
        ]
    })
    
    # Print the result
    print("\nAgent response after optimization:")
    print("-" * 50)
    if "messages" in result and result["messages"]:
        for message in result["messages"]:
            if isinstance(message, dict) and "role" in message and message["role"] == "assistant":
                print(message["content"])
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_memory_optimization()) 