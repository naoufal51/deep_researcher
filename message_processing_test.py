"""
Simple test script to verify message processing logic.

This script tests that our message handling code can correctly extract content
from different response formats that might be returned by agents.
"""

import asyncio
from typing import Any, Dict, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Import our message handling function
from src.react_agent.graph import (
    get_user_message, 
    initialize_state_if_needed,
    add_message_to_state
)

# Define test cases for different message formats
TEST_CASES = [
    {
        "name": "Dictionary with content",
        "message": {"role": "assistant", "content": "This is a test message"},
        "expected": "This is a test message"
    },
    {
        "name": "Simple string",
        "message": "This is a simple string message",
        "expected": "This is a simple string message"
    },
    {
        "name": "Dictionary with text field",
        "message": {"text": "This is in the text field"},
        "expected": "This is in the text field"
    },
    {
        "name": "AIMessage object",
        "message": AIMessage(content="This is from an AIMessage"),
        "expected": "This is from an AIMessage"
    },
    {
        "name": "HumanMessage object", 
        "message": HumanMessage(content="This is from a HumanMessage"),
        "expected": "This is from a HumanMessage"
    },
    {
        "name": "Messages array in response",
        "message": {"messages": [{"role": "user", "content": "First message"}, {"role": "assistant", "content": "Last message"}]},
        "expected": "Last message"
    },
    {
        "name": "Messages array with AIMessage objects",
        "message": {"messages": [HumanMessage(content="User message"), AIMessage(content="AI response")]},
        "expected": "AI response"
    }
]

# Helper functions to simulate agent responses
def simulate_direct_message_response(content: str) -> AIMessage:
    """Simulate a direct AIMessage response."""
    return AIMessage(content=content)

def simulate_dict_response(content: str) -> Dict[str, Any]:
    """Simulate a dictionary response with content field."""
    return {"content": content}

def simulate_messages_response(content: str) -> Dict[str, List[Dict[str, str]]]:
    """Simulate a dictionary with messages array."""
    return {
        "messages": [
            {"role": "user", "content": "User query"},
            {"role": "assistant", "content": content}
        ]
    }

def simulate_messages_with_objects(content: str) -> Dict[str, List[Any]]:
    """Simulate a dictionary with messages as objects."""
    return {
        "messages": [
            HumanMessage(content="User query"),
            AIMessage(content=content)
        ]
    }

def test_get_user_message():
    """Test the get_user_message function with different formats."""
    print("\n=== Testing get_user_message function ===")
    
    for test_case in TEST_CASES:
        state = {"messages": [test_case["message"]]}
        result = get_user_message(state)
        
        success = result == test_case["expected"]
        status = "✅ PASS" if success else f"❌ FAIL - Got: '{result}'"
        print(f"{test_case['name']}: {status}")

def test_add_message_to_state():
    """Test adding messages to state."""
    print("\n=== Testing add_message_to_state function ===")
    
    # Test adding a message to empty state
    state = {}
    result = add_message_to_state(state, "Test message")
    success = len(state["messages"]) == 1 and state["messages"][0]["content"] == "Test message"
    print(f"Add to empty state: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Test adding with custom role
    state = {"messages": []}
    result = add_message_to_state(state, "User message", "user")
    success = state["messages"][0]["role"] == "user"
    print(f"Add with custom role: {'✅ PASS' if success else '❌ FAIL'}")

def test_message_extraction():
    """Test extraction of content from different response formats."""
    print("\n=== Testing message extraction from different response formats ===")
    
    test_cases = [
        {
            "name": "Direct AIMessage",
            "response": simulate_direct_message_response("Test content"),
            "expected": "Test content"
        },
        {
            "name": "Dictionary with content",
            "response": simulate_dict_response("Test content"),
            "expected": "Test content"
        },
        {
            "name": "Dictionary with messages array",
            "response": simulate_messages_response("Test content"),
            "expected": "Test content"
        },
        {
            "name": "Dictionary with message objects",
            "response": simulate_messages_with_objects("Test content"),
            "expected": "Test content"
        }
    ]
    
    for test in test_cases:
        response = test["response"]
        
        # Apply the same extraction logic we use in our agent functions
        if hasattr(response, 'content'):  # It's a Message object
            result = response.content
        elif isinstance(response, dict) and "content" in response:
            result = response["content"]
        elif isinstance(response, dict) and "messages" in response:
            # Get the last message
            last_message = response["messages"][-1]
            # Handle if the last message is a dict or Message object
            if hasattr(last_message, 'content'):
                result = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                result = last_message["content"]
            else:
                result = str(last_message)
        else:
            # Fallback to string representation
            result = str(response)
        
        success = result == test["expected"]
        status = "✅ PASS" if success else f"❌ FAIL - Got: '{result}'"
        print(f"{test['name']}: {status}")

if __name__ == "__main__":
    # Run the tests
    test_get_user_message()
    test_add_message_to_state()
    test_message_extraction()
    
    print("\nAll tests completed.") 