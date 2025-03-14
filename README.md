# DeepResearcher: AI-Powered Advanced Research and Writing System

[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-blue)](https://langchain-ai.github.io/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)](https://github.com/naoufal51/deep_researcher)

> **Note:** This project is currently under active development. Features and documentation may change frequently.

DeepResearcher is a sophisticated multi-agent system built with LangGraph that automates the research and content creation process. By orchestrating specialized AI agents working together through a defined workflow, it transforms queries into comprehensive, high-quality content with minimal human intervention.

## Key Features

### 🧠 Multi-Agent Architecture
- **Research Agent**: Finds accurate information from multiple sources including Tavily, ArXiv, and Wikipedia 
- **Math Agent**: Performs calculations and mathematical analysis when needed
- **Writing Agent**: Transforms research findings into well-written, coherent content
- **Publishing Agent**: Polishes and formats content to meet publication standards

### 🔄 Intelligent Workflow
- **Dynamically Routed Workflow**: Automatically determines which agents to engage based on query analysis
- **Safety Mechanisms**: Content screening for harmful material
- **Quality Verification**: Multiple verification and revision steps ensure high-quality output
- **Self-Improvement**: Procedural memory optimization based on feedback

### 🛠️ Advanced Features
- **Semantic Queries**: Generates optimal search queries based on user input
- **Fact Verification**: Ensures output accuracy through multiple verification passes
- **Content Revision**: Automatically improves content based on quality assessments
- **Tool Usage Verification**: Ensures proper processing of all tool outputs

### 📊 Comprehensive Testing
- **Component Testing**: Verify each agent and workflow component functions correctly
- **End-to-End Evaluation**: Test complete user experiences
- **LLM-Based Quality Assessment**: Automatic evaluation of content quality

## Architecture

DeepResearcher uses a state-based workflow architecture with specialized agents:

```
User Query → Safety Check → Research Team (Research + Math) → Tool Verification → Writing Team → Quality Reflection → Final Output
```

Each agent has:
- Specialized prompts stored in procedural memory
- Access to relevant tools (search, calculators, etc.)
- Clear responsibilities within the workflow

## Usage

### Prerequisites
- Python 3.9+
- OpenAI API key (required)
- Tavily API key (for search functionality)
- LangChain API key (optional, for run tracking)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deep_researcher.git
cd deep_researcher

# Install dependencies
pip install uv
uv sync --dev
```

### Running the Application

```bash
# Start the LangGraph development server
langgraph dev --no-browser

# Or using the provided script
./run_tests.sh dev
```

Then visit `http://localhost:2024` to interact with DeepResearcher.

### Running Tests

```bash
# Run agent evaluations
./run_tests.sh eval

# Or directly with Python
python run_evals.py
```

## Customization

### Modifying Agent Prompts

Agent prompts are stored in a procedural memory system and can be optimized over time based on feedback:

```python
# Example: Updating the research agent prompt
memory_store.put(
    (("instructions",)), 
    key="research_agent", 
    value={
        "prompt": "Your custom research agent prompt here..."
    }
)
```

### Adding New Tools

Extend DeepResearcher's capabilities by adding custom tools to relevant agents:

```python
# Example: Adding a new tool to the research agent
@tool
def custom_research_tool(query: str) -> str:
    """Description of what the tool does"""
    # Tool implementation
    return result

# Add to research tools list
research_tools.append(custom_research_tool)
```

### Workflow Customization

Modify the workflow by adjusting the StateGraph configuration:

```python
# Example: Adding a new node to the research workflow
research_workflow.add_node("my_custom_node", my_custom_function)
research_workflow.add_edge("research_agent", "my_custom_node")
research_workflow.add_edge("my_custom_node", "math_agent")
```

## Technical Details

DeepResearcher is built using:

- **LangGraph**: For defining agent workflows and state management
- **LangChain**: For tool integration and LLM interaction
- **OpenAI**: For powerful LLM capabilities (default is gpt-4o-mini)
- **Fast API / FastHTML**: For web interface (when applicable)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request