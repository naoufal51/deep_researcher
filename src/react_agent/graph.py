"""A structured research and writing system using LangGraph."""

from typing import Dict, List, Any, Optional, Literal, TypeVar, Union, cast
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from openevals.llm import create_llm_as_judge

# Initialize memory store for procedural memory
memory_store = InMemoryStore()

# Initialize with default prompts if not already there
try:
    memory_store.get((("instructions",)), key="research_agent")
except KeyError:
    memory_store.put(
        (("instructions",)), 
        key="research_agent", 
        value={
            "prompt": "You are a research specialist. Your task is to find accurate information on any topic. Use search tools to gather relevant data. Be thorough and precise in your research. IMPORTANT: When using search tools, ALWAYS process the raw results into a coherent summary. NEVER return raw JSON, URLs, or unprocessed search results directly. Always convert tool outputs into human-readable information."
        }
    )

try:
    memory_store.get((("instructions",)), key="writing_agent")
except KeyError:
    memory_store.put(
        (("instructions",)), 
        key="writing_agent", 
        value={
            "prompt": "You are a writing specialist. Your job is to create well-written, clear, and engaging content based on provided information. Focus on quality writing, proper structure, and effective communication. When given research information, transform it into well-structured, coherent content. Never include raw data, URLs, or JSON in your output."
        }
    )

try:
    memory_store.get((("instructions",)), key="publishing_agent")
except KeyError:
    memory_store.put(
        (("instructions",)), 
        key="publishing_agent", 
        value={
            "prompt": "You are a publishing specialist. Your task is to format, polish, and finalize content to meet publication standards. Focus on layout, consistency, and professional presentation. Take draft content and improve its formatting, structure, and overall readability. Your output will be the final result presented to the user, so make it perfect. NEVER include raw data, URLs, JSON, or any meta-commentary in your output - provide ONLY the polished article content itself. Remove any traces of raw data or search results from your final output."
        }
    )

try:
    memory_store.get((("instructions",)), key="math_agent")
except KeyError:
    memory_store.put(
        (("instructions",)), 
        key="math_agent", 
        value={
            "prompt": "You are a mathematics specialist. Your role is to perform calculations, solve math problems, and provide numerical analysis. Use the math_calculation tool for performing calculations. Always present your results in a clear, human-readable format."
        }
    )

# Define state types
class SearchQuery(BaseModel):
    """A search query with explanation."""
    search_query: str = Field(description="The search query to execute")
    explanation: str = Field(description="Explanation of why this query is relevant")

class SearchQueries(BaseModel):
    """Collection of search queries."""
    queries: List[SearchQuery] = Field(description="List of search queries")

class ResearchState(TypedDict):
    """State for the research process."""
    query: str
    search_queries: NotRequired[List[SearchQuery]]
    research_results: NotRequired[str]

class WritingState(TypedDict):
    """State for the writing process."""
    research_findings: str
    final_content: NotRequired[str]
    revision_history: NotRequired[List[Dict[str, str]]]

class AgentFeedback(TypedDict):
    """Feedback for improving agent prompts."""
    agent_name: str
    feedback: str
    trajectory: List[Dict[str, Any]]

class MainState(TypedDict):
    """Main workflow state."""
    messages: List[Union[Dict[str, Any], str]]
    research_state: NotRequired[ResearchState]
    writing_state: NotRequired[WritingState]
    feedback: NotRequired[AgentFeedback]  # For tracking feedback to optimize prompts

class MainStateInput(TypedDict):
    """Input state schema."""
    messages: List[Union[Dict[str, Any], str]]

class MainStateOutput(TypedDict):
    """Output state schema."""
    messages: List[Union[Dict[str, Any], str]]
    research_state: NotRequired[ResearchState]
    writing_state: NotRequired[WritingState]

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tavily_tool = TavilySearchResults(max_results=2)

# Initialize the model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Tool function
@tool
def math_calculation(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.
    Only use this for simple math calculations. Do not use for code execution.
    Examples: '2+2', '5*10', 'sqrt(4)', 'sin(0)'
    """
    try:
        # Very limited evaluation for basic math only
        import math
        
        # Create a safe namespace with only math functions
        safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        safe_dict.update({
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum
        })
        
        # Evaluate the expression in the safe namespace
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# Set up tools for agents
research_tools = [tavily_tool, arxiv_tool, wiki_tool]
math_tools = [math_calculation]
writing_tools = []
publishing_tools = []

# Procedural memory prompt functions
def get_research_agent_prompt(state):
    """Get research agent prompt from memory store."""
    try:
        item = memory_store.get((("instructions",)), key="research_agent")
        instructions = item.value["prompt"]
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']
    except Exception as e:
        # Fallback to default prompt if there's an error
        instructions = "You are a research specialist. Your task is to find accurate information on any topic."
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']

def get_math_agent_prompt(state):
    """Get math agent prompt from memory store."""
    try:
        item = memory_store.get((("instructions",)), key="math_agent")
        instructions = item.value["prompt"]
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']
    except Exception as e:
        # Fallback to default prompt if there's an error
        instructions = "You are a mathematics specialist. Your role is to perform calculations."
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']

def get_writing_agent_prompt(state):
    """Get writing agent prompt from memory store."""
    try:
        item = memory_store.get((("instructions",)), key="writing_agent")
        instructions = item.value["prompt"]
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']
    except Exception as e:
        # Fallback to default prompt if there's an error
        instructions = "You are a writing specialist. Your job is to create well-written content."
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']

def get_publishing_agent_prompt(state):
    """Get publishing agent prompt from memory store."""
    try:
        item = memory_store.get((("instructions",)), key="publishing_agent")
        instructions = item.value["prompt"]
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']
    except Exception as e:
        # Fallback to default prompt if there's an error
        instructions = "You are a publishing specialist. Your task is to format and polish content."
        sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
        return [sys_prompt] + state['messages']

# Create the research team agents with procedural memory
research_agent = create_react_agent(
    model,
    research_tools,
    prompt=get_research_agent_prompt,
    name="research_agent"
)

math_agent = create_react_agent(
    model,
    math_tools,
    prompt=get_math_agent_prompt,
    name="math_agent"
)

# Create the writing team agents with procedural memory
writing_agent = create_react_agent(
    model,
    writing_tools,
    prompt=get_writing_agent_prompt,
    name="writing_agent"
)

publishing_agent = create_react_agent(
    model,
    publishing_tools,
    prompt=get_publishing_agent_prompt,
    name="publishing_agent"
)

# Helper functions for message handling and state management
def get_user_message(state: MainState) -> str:
    """Safely extract the last user message from state."""
    messages = state.get("messages", [])
    if not messages:
        return ""
    
    # Find the last message
    last_message = messages[-1]
    
    # Handle different message formats
    if isinstance(last_message, dict):
        if "content" in last_message:
            return last_message["content"]
        elif "text" in last_message:
            return last_message["text"]
        elif "messages" in last_message:
            # If the message contains a messages array, get the last one
            sub_messages = last_message["messages"]
            if sub_messages and len(sub_messages) > 0:
                sub_last = sub_messages[-1]
                if isinstance(sub_last, dict) and "content" in sub_last:
                    return sub_last["content"]
                elif hasattr(sub_last, 'content'):
                    return sub_last.content
    elif isinstance(last_message, str):
        return last_message
    elif hasattr(last_message, 'content'):  # Message object like AIMessage
        return last_message.content
    elif hasattr(last_message, 'text'):
        return last_message.text
    
    # Default fallback
    return str(last_message)

def initialize_state_if_needed(state: MainState) -> MainState:
    """Initialize state dictionaries if they don't exist yet."""
    # Ensure messages list exists
    if "messages" not in state:
        state["messages"] = []
    
    return state

def add_message_to_state(state: MainState, content: str, role: str = "assistant") -> MainState:
    """Add a message to the state with proper formatting."""
    if "messages" not in state:
        state["messages"] = []
    
    state["messages"].append({"role": role, "content": content})
    return state

# Research Team Functions
async def generate_search_queries(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Generate search queries based on the user query."""
    state = initialize_state_if_needed(state)
    user_message = get_user_message(state)
    
    # Create system prompt for query generation
    system_prompt = """You are a research specialist who creates effective search queries.
    Given a user's request, generate 2-3 search queries that will help gather comprehensive information.
    Your queries should be diverse and cover different aspects of the topic.
    NEVER return raw JSON, URLs, or meta-commentary in your output.
    """
    
    # Generate queries using LLM with structured output
    structured_llm = model.with_structured_output(SearchQueries)
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate search queries to research: {user_message}")
    ])
    
    # Initialize research state if needed
    if "research_state" not in state:
        state["research_state"] = {"query": user_message}
    
    # Update state with queries
    state["research_state"]["search_queries"] = result.queries
    
    return state

async def run_research_agent(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Run the research agent to gather information."""
    state = initialize_state_if_needed(state)
    user_message = get_user_message(state)
    
    # Check if we already have search queries
    if "research_state" not in state or "search_queries" not in state["research_state"]:
        # Generate search queries first
        state = await generate_search_queries(state, config)
    
    # Format the search queries as a prompt for the research agent
    search_queries = state["research_state"].get("search_queries", [])
    query_text = "\n".join([f"- {q.search_query}: {q.explanation}" for q in search_queries])
    
    research_prompt = f"""Please research the following topic thoroughly:
    
    User request: {user_message}
    
    Use these search queries to guide your research:
    {query_text}
    
    Provide a comprehensive, well-organized summary of your findings.
    """
    
    # Get response from research agent
    response = research_agent.invoke({"messages": [{"role": "user", "content": research_prompt}]})
    
    # Track trajectory for potential feedback
    agent_trajectory = []
    if isinstance(response, dict) and "messages" in response:
        agent_trajectory = response["messages"]
    
    # Handle different response formats
    if hasattr(response, 'content'):  # It's a Message object
        result = response.content
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
    
    # Update state
    state["research_state"]["research_results"] = result
    
    # Store trajectory for potential feedback later
    if "feedback" not in state:
        state["feedback"] = {"agent_name": "research_agent", "trajectory": agent_trajectory, "feedback": ""}
    
    return state

async def run_math_agent(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Run the math agent to perform calculations if needed."""
    state = initialize_state_if_needed(state)
    user_message = get_user_message(state)
    
    # Ensure research results exist
    if "research_state" not in state or "research_results" not in state["research_state"]:
        # Need to run research first
        state = await run_research_agent(state, config)
    
    research_results = state["research_state"]["research_results"]
    
    # Check if calculations are needed
    calculation_prompt = f"""Based on the research results, determine if mathematical analysis is needed:
    User request: {user_message}
    Research findings: {research_results}
    
    If calculations are needed, formulate the exact calculation needed.
    If no calculations are needed, respond with 'No calculations needed'.
    """
    
    calculation_decision = model.invoke([SystemMessage(content="You decide if mathematical calculations are needed."), 
                                      HumanMessage(content=calculation_prompt)]).content
    
    if "no calculations needed" not in calculation_decision.lower():
        # Run math agent
        response = math_agent.invoke({"messages": [{"role": "user", "content": calculation_decision}]})
        
        # Handle different response formats
        if hasattr(response, 'content'):  # It's a Message object
            math_result = response.content
        elif isinstance(response, dict) and "messages" in response:
            # Get the last message
            last_message = response["messages"][-1]
            # Handle if the last message is a dict or Message object
            if hasattr(last_message, 'content'):
                math_result = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                math_result = last_message["content"]
            else:
                math_result = str(last_message)
        else:
            # Fallback to string representation
            math_result = str(response)
        
        # Append math results to research results
        state["research_state"]["research_results"] += f"\n\nMathematical Analysis:\n{math_result}"
    
    return state

def analyze_research_needs(state: MainState) -> str:
    """Determine which research agent to use."""
    state = initialize_state_if_needed(cast(MainState, state))
    user_message = get_user_message(state)
    
    # Analyze the query to determine if it needs math
    analysis_prompt = f"""Analyze this user request and determine if it requires:
    1. General research (searching for information)
    2. Mathematical calculations
    3. Both research and calculations
    
    User request: {user_message}
    
    Respond with one of: 'research_only', 'math_only', or 'both'.
    """
    
    analysis = model.invoke([SystemMessage(content="You analyze user requests to determine needed expertise."), 
                          HumanMessage(content=analysis_prompt)]).content
    
    if "both" in analysis.lower():
        return "both"
    elif "math" in analysis.lower():
        return "math_only"
    else:
        return "research_only"

# Helper functions for conditional routing
def should_go_to_math_agent(state: MainState) -> bool:
    """Determine if we should go to math agent after research agent."""
    return analyze_research_needs(state) == "both"

# Writing Team Functions
async def run_writing_agent(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Run the writing agent to create content based on research."""
    state = initialize_state_if_needed(state)
    user_message = get_user_message(state)
    
    # Get research results
    research_results = state["research_state"].get("research_results", "")
    
    writing_prompt = f"""Based on the following research, please write a comprehensive, well-structured article:
    
    User request: {user_message}
    
    Research findings:
    {research_results}
    
    Your article should be informative, engaging, and well-organized with appropriate headings and subheadings.
    """
    
    # Get response from writing agent
    response = writing_agent.invoke({"messages": [{"role": "user", "content": writing_prompt}]})
    
    # Track trajectory for potential feedback
    agent_trajectory = []
    if isinstance(response, dict) and "messages" in response:
        agent_trajectory = response["messages"]
    
    # Handle different response formats
    if hasattr(response, 'content'):  # It's a Message object
        result = response.content
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
    
    # Update state
    if "writing_state" not in state:
        state["writing_state"] = {}
    state["writing_state"]["final_content"] = result
    
    # Store trajectory for potential feedback later
    if "feedback" not in state:
        state["feedback"] = {"agent_name": "writing_agent", "trajectory": agent_trajectory, "feedback": ""}
    elif state["feedback"]["agent_name"] == "research_agent":
        # If we already have research feedback, create a new entry for writing feedback
        state["writing_feedback"] = {"agent_name": "writing_agent", "trajectory": agent_trajectory, "feedback": ""}
    
    return state

async def run_publishing_agent(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Run the publishing agent to polish the content."""
    state = initialize_state_if_needed(state)
    user_message = get_user_message(state)
    
    # Ensure writing results exist
    if "writing_state" not in state or "final_content" not in state["writing_state"]:
        # If no writing results, need to run the writing agent first
        state = await run_writing_agent(state, config)
    
    draft_content = state["writing_state"]["final_content"]
    
    # Prepare prompt for publishing
    publishing_prompt = f"""Polish and finalize this draft content to professional standards:
    
    User request: {user_message}
    
    Draft content: {draft_content}
    
    Improve formatting, structure, and readability. Your output will be presented directly to the user,
    so ensure it contains ONLY the polished article with no raw data, meta-commentary, or introductory text.
    """
    
    # Get response from publishing agent
    response = publishing_agent.invoke({"messages": [{"role": "user", "content": publishing_prompt}]})
    
    # Handle different response formats
    if hasattr(response, 'content'):  # It's a Message object
        final_content = response.content
    elif isinstance(response, dict) and "messages" in response:
        # Get the last message
        last_message = response["messages"][-1]
        # Handle if the last message is a dict or Message object
        if hasattr(last_message, 'content'):
            final_content = last_message.content
        elif isinstance(last_message, dict) and "content" in last_message:
            final_content = last_message["content"]
        else:
            final_content = str(last_message)
    else:
        # Fallback to string representation
        final_content = str(response)
    
    # Update state with final content
    state["writing_state"]["final_content"] = final_content
    
    # Add final response to messages
    state = add_message_to_state(state, final_content)
    
    return state

def assess_content_completeness(state: MainState) -> str:
    """Assess if the content is complete or needs more research."""
    if "writing_state" not in state or "final_content" not in state["writing_state"]:
        return "needs_more_research"
    
    final_content = state["writing_state"]["final_content"]
    original_query = get_user_message(state)
    
    assessment_prompt = f"""
    Original user question: {original_query}
    
    Final content:
    {final_content}
    
    Does this content completely address the user's needs, or is more research required?
    Respond with either 'complete' or 'needs_more_research'.
    """
    
    # Get assessment
    assessment = model.invoke([SystemMessage(content="You assess content completeness."), 
                            HumanMessage(content=assessment_prompt)]).content
    
    if "needs_more_research" in assessment.lower():
        return "needs_more_research"
    else:
        return "complete"

# New functions to enhance the system

def assess_query_complexity(state: MainState) -> str:
    """Assess the complexity of the query to determine processing approach.
    
    This function analyzes the user query to determine how complex it is,
    which helps in planning the appropriate research and writing strategy.
    """
    user_message = get_user_message(state)
    
    complexity_prompt = f"""Analyze this user query and determine its complexity level:
    
    Query: {user_message}
    
    Categorize as one of:
    - "simple" (straightforward factual question)
    - "moderate" (requires research but no complex analysis)
    - "complex" (requires deep research and synthesis)
    
    Respond with ONLY the category word.
    """
    
    complexity = model.invoke([
        SystemMessage(content="You analyze query complexity."),
        HumanMessage(content=complexity_prompt)
    ]).content.strip().lower()
    
    if "complex" in complexity:
        return "complex"
    elif "moderate" in complexity:
        return "moderate"
    else:
        return "simple"

# Add a helper function to handle the safety check result
def get_safety_status(state: MainState) -> str:
    """Extract the safety status for routing."""
    # If the state has already been processed by check_safety
    if "safety_status" in state and state["safety_status"] in ["safe", "unsafe"]:
        return state["safety_status"]
    return "safe"  # Default to safe if not set

# Modify the check_safety function to store the status in state
def check_safety(state: MainState) -> MainState:
    """Check content for safety and compliance issues.
    
    This function evaluates user queries for potentially unsafe content 
    and routes them appropriately.
    """
    content = get_user_message(state)
    
    safety_prompt = f"""Evaluate if this content contains any unsafe or harmful elements:
    
    Content: {content}
    
    Check for:
    1. Harmful instructions
    2. Illegal content requests
    3. Hateful, harassing, or violent content
    4. Adult or explicit content
    5. Personal or private information
    
    If ANY issues are found, respond with "UNSAFE: [reason]".
    If content is safe, respond with "SAFE".
    """
    
    safety_check = model.invoke([
        SystemMessage(content="You are a safety evaluation system."),
        HumanMessage(content=safety_prompt)
    ]).content
    
    if "UNSAFE" in safety_check:
        # If unsafe, add a safety message to state and set flag to skip processing
        state = add_message_to_state(
            state,
            "I'm unable to process this request as it appears to contain or request content that violates usage guidelines or policies.",
            "assistant"
        )
        state["safety_status"] = "unsafe"
    else:
        state["safety_status"] = "safe"
    
    return state

async def verify_content_quality(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Evaluate the content quality and request improvements if needed.
    
    This function acts as a self-reflection mechanism to ensure the 
    generated content meets high quality standards before delivery.
    """
    if "writing_state" not in state or "final_content" not in state["writing_state"]:
        return state
    
    final_content = state["writing_state"]["final_content"]
    user_query = get_user_message(state)
    
    # Create a structured critique prompt similar to the reflection example
    critique_prompt = f"""You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response to the user query.

    User query: "{user_query}"
    
    AI response:
    {final_content}
    
    Evaluate the response based on these criteria:
    1. Accuracy - Is the information correct and factual?
    2. Completeness - Does it fully address the user's query?
    3. Clarity - Is the explanation clear and well-structured?
    4. Helpfulness - Does it provide actionable information?
    5. Safety - Does it avoid harmful content?
    
    If the response meets ALL criteria satisfactorily, set pass to True.
    If you find ANY issues with the response, provide specific and constructive feedback in the comment key and set pass to False.
    """
    
    # Use the LLM-as-judge approach from openevals
    evaluator = create_llm_as_judge(
        prompt=critique_prompt,
        model="openai:gpt-4o-mini",
        feedback_key="pass",
    )
    
    eval_result = evaluator(outputs=final_content, inputs=user_query)
    
    if eval_result["score"]:
        # Content passed evaluation
        return state
    else:
        # Content needs revision
        return await revise_content(state, eval_result["comment"], config)

async def revise_content(state: MainState, feedback: str, config: Optional[RunnableConfig] = None) -> MainState:
    """Revise content based on quality feedback.
    
    This function handles improving content when the reflection system
    identifies areas for enhancement.
    """
    if "writing_state" not in state:
        return state
    
    current_content = state["writing_state"]["final_content"]
    user_query = get_user_message(state)
    
    revision_prompt = f"""Revise this content based on the following feedback:
    
    Original query: {user_query}
    
    Current content:
    {current_content}
    
    Feedback from evaluator:
    {feedback}
    
    Provide a complete revised version that addresses all feedback points.
    """
    
    # Get revised content
    revised_content = model.invoke([
        SystemMessage(content="You are an expert content reviser."),
        HumanMessage(content=revision_prompt)
    ]).content
    
    # Update state with revised content
    state["writing_state"]["final_content"] = revised_content
    state["writing_state"]["revision_history"] = state["writing_state"].get("revision_history", []) + [
        {"original": current_content, "feedback": feedback, "revised": revised_content}
    ]
    
    # Update the message in the state
    # Replace the last message (which would be the unrevised content)
    if state["messages"] and len(state["messages"]) > 0:
        state["messages"][-1] = {"role": "assistant", "content": revised_content}
    else:
        state = add_message_to_state(state, revised_content)
    
    return state

async def verify_tool_usage(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Verify that tools were used correctly and data is processed.
    
    This function checks research results for unprocessed data and cleans it
    if necessary, ensuring high-quality content.
    """
    if "research_state" not in state or "research_results" not in state["research_state"]:
        return state
    
    research_results = state["research_state"]["research_results"]
    
    # Check for signs of raw/unprocessed data
    raw_data_indicators = [
        "tavily_search_results", 
        "\"urls\":", 
        "\"content\":", 
        "arxiv.org/pdf/", 
        "Page:", 
        "\"title\":",
        "JSON"
    ]
    
    needs_processing = any(indicator in research_results for indicator in raw_data_indicators)
    
    if needs_processing:
        clean_prompt = f"""There appears to be raw or unprocessed data in these research results:
        
        {research_results}
        
        Please transform ALL raw data into a coherent, human-readable summary.
        Remove ALL JSON structures, URLs, and similar raw elements.
        Present only polished, processed information.
        """
        
        processed_results = model.invoke([
            SystemMessage(content="You are a data processor who transforms raw data into readable content."),
            HumanMessage(content=clean_prompt)
        ]).content
        
        # Update with processed results
        state["research_state"]["research_results"] = processed_results
    
    return state

# Define the workflow components

# Define the research workflow
research_workflow = StateGraph(MainState)
research_workflow.add_node("research_agent", run_research_agent)
research_workflow.add_node("math_agent", run_math_agent)

# Add research workflow edges
research_workflow.add_conditional_edges(
    START,
    analyze_research_needs,
    {
        "research_only": "research_agent",
        "math_only": "math_agent",
        "both": "research_agent"
    }
)

# Add conditional edges for after research agent
research_workflow.add_conditional_edges(
    "research_agent",
    should_go_to_math_agent,
    {
        True: "math_agent",
        False: END
    }
)

# Add edge for math agent to end
research_workflow.add_edge("math_agent", END)

# Define the writing workflow
writing_workflow = StateGraph(MainState)
writing_workflow.add_node("writing_agent", run_writing_agent)
writing_workflow.add_node("publishing_agent", run_publishing_agent)

# Add writing workflow edges
writing_workflow.add_edge(START, "writing_agent")
writing_workflow.add_edge("writing_agent", "publishing_agent")
writing_workflow.add_edge("publishing_agent", END)

# Define the planning workflow
planning_workflow = StateGraph(MainState)
planning_workflow.add_node("check_safety", check_safety)

# Update planning workflow edges
planning_workflow.add_edge(START, "check_safety")
planning_workflow.add_conditional_edges(
    "check_safety",
    get_safety_status,
    {
        "unsafe": END,
        "safe": END
    }
)

# Define reflection workflow
reflection_workflow = StateGraph(MainState)
reflection_workflow.add_node("verify_content_quality", verify_content_quality)
reflection_workflow.add_edge(START, "verify_content_quality")
reflection_workflow.add_edge("verify_content_quality", END)

# Define tool verification workflow
tool_verification_workflow = StateGraph(MainState)
tool_verification_workflow.add_node("verify_tool_usage", verify_tool_usage)
tool_verification_workflow.add_edge(START, "verify_tool_usage")
tool_verification_workflow.add_edge("verify_tool_usage", END)

# Create the main workflow with the enhanced components
main_workflow = StateGraph(MainState, input=MainStateInput, output=MainStateOutput)
main_workflow.add_node("planning", planning_workflow.compile())
main_workflow.add_node("research_team", research_workflow.compile())
main_workflow.add_node("tool_verification", tool_verification_workflow.compile())
main_workflow.add_node("writing_team", writing_workflow.compile())
main_workflow.add_node("reflection", reflection_workflow.compile())

# Add main workflow edges with safety check
main_workflow.add_edge(START, "planning")
main_workflow.add_conditional_edges(
    "planning",
    get_safety_status,
    {
        "unsafe": END,  # Skip further processing if content is unsafe
        "safe": "research_team"  # Continue to research if content is safe
    }
)
main_workflow.add_edge("research_team", "tool_verification")
main_workflow.add_edge("tool_verification", "writing_team")
main_workflow.add_edge("writing_team", "reflection")
main_workflow.add_conditional_edges(
    "reflection", 
    assess_content_completeness,
    {
        "needs_more_research": "research_team",
        "complete": END
    }
)

# Set up checkpointing for the graph
checkpointer = MemorySaver()

# Compile the graph
graph = main_workflow.compile(checkpointer=checkpointer)

# New functions for procedural memory optimization

async def process_feedback(state: MainState, config: Optional[RunnableConfig] = None) -> MainState:
    """Process user feedback to optimize agent prompts."""
    state = initialize_state_if_needed(state)
    
    if "feedback" not in state or not state["feedback"]["feedback"]:
        # No feedback to process
        return state
    
    feedback_data = state["feedback"]
    agent_name = feedback_data["agent_name"]
    trajectory = feedback_data["trajectory"]
    feedback = feedback_data["feedback"]
    
    try:
        # Load the existing prompt
        item = memory_store.get((("instructions",)), key=agent_name)
        current_prompt = item.value["prompt"]
        
        # Format trajectory for prompt optimization
        formatted_trajectory = []
        for msg in trajectory:
            if isinstance(msg, dict):
                formatted_trajectory.append(msg)
            elif hasattr(msg, 'to_dict'):
                formatted_trajectory.append(msg.to_dict())
            else:
                # Skip messages we can't format properly
                continue
        
        # Create a simple prompt optimizer function (similar to langmem but without the dependency)
        optimization_prompt = f"""You are an expert at improving AI system prompts based on user feedback.

        Current prompt:
        {current_prompt}

        User feedback on agent performance:
        {feedback}

        Analyze the feedback and modify the prompt to address the issues while maintaining its core functionality.
        Return ONLY the new improved prompt text with no additional commentary.
        """
        
        # Get optimized prompt
        optimized_prompt = model.invoke([
            SystemMessage(content="You improve system prompts based on user feedback."),
            HumanMessage(content=optimization_prompt)
        ]).content
        
        # Update the prompt in memory store
        memory_store.put(
            (("instructions",)),
            key=agent_name,
            value={"prompt": optimized_prompt}
        )
        
        # Clear feedback after processing
        state["feedback"] = {"agent_name": "", "trajectory": [], "feedback": ""}
        
    except Exception as e:
        # Handle errors gracefully
        print(f"Error optimizing prompt: {str(e)}")
    
    return state
