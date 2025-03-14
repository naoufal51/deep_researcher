#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}  LangGraph Agent Testing Suite${NC}"
echo -e "${BLUE}====================================${NC}"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check environment variables
check_environment() {
  local missing=false
  
  echo -e "${YELLOW}Checking environment...${NC}"
  
  if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ OPENAI_API_KEY is not set${NC}"
    missing=true
  else
    echo -e "${GREEN}✓ OPENAI_API_KEY is set${NC}"
  fi
  
  # Check for optional LangSmith integration
  if [ -z "$LANGCHAIN_API_KEY" ]; then
    echo -e "${YELLOW}⚠️ LANGCHAIN_API_KEY is not set (optional for LangSmith)${NC}"
  else
    echo -e "${GREEN}✓ LANGCHAIN_API_KEY is set${NC}"
    
    if [ -z "$LANGCHAIN_PROJECT" ]; then
      echo -e "${YELLOW}⚠️ LANGCHAIN_PROJECT is not set - using default project${NC}"
    else
      echo -e "${GREEN}✓ LANGCHAIN_PROJECT is set to: $LANGCHAIN_PROJECT${NC}"
    fi
  fi
  
  # Check for virtual environment
  if [ -f ".venv/bin/python" ]; then
    PYTHON_EXECUTABLE=".venv/bin/python"
    echo -e "${GREEN}✓ Using Python from local .venv${NC}"
  elif [ -f "venv/bin/python" ]; then
    PYTHON_EXECUTABLE="venv/bin/python"
    echo -e "${GREEN}✓ Using Python from local venv${NC}"
  elif command_exists uv; then
    PYTHON_EXECUTABLE="uv run python"
    echo -e "${YELLOW}⚠️ No local venv found, using uv run${NC}"
  elif command_exists python; then
    PYTHON_EXECUTABLE="python"
    echo -e "${YELLOW}⚠️ No local venv found, using system Python${NC}"
  else
    echo -e "${RED}❌ No Python interpreter found${NC}"
    missing=true
  fi
  
  if [ "$missing" = true ]; then
    echo -e "${RED}Please set the required environment variables and ensure Python is installed.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Environment check passed!${NC}"
  echo ""
}

# Function to run evaluations
run_evaluations() {
  echo -e "${BLUE}Running agent evaluations...${NC}"
  
  if [ "$PYTHON_EXECUTABLE" = "uv run python" ]; then
    uv run python run_evals.py
  else
    $PYTHON_EXECUTABLE run_evals.py
  fi
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluations completed successfully${NC}"
  else
    echo -e "${RED}❌ Evaluations failed${NC}"
  fi
}

# Function to run the LangGraph development server
run_langgraph() {
  echo -e "${BLUE}Starting LangGraph development server...${NC}"
  
  if [ -f ".venv/bin/langgraph" ]; then
    .venv/bin/langgraph dev --no-browser
  elif [ -f "venv/bin/langgraph" ]; then
    venv/bin/langgraph dev --no-browser
  elif [ "$PYTHON_EXECUTABLE" = "uv run python" ]; then
    uv run langgraph dev --no-browser
  else
    echo -e "${YELLOW}Attempting to run langgraph with system Python...${NC}"
    langgraph dev --no-browser
  fi
}

# Main script logic
check_environment

# Parse command line arguments
case "$1" in
  "eval")
    run_evaluations
    ;;
  "dev")
    run_langgraph
    ;;
  *)
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  ${GREEN}./run_tests.sh eval${NC} - Run agent evaluations"
    echo -e "  ${GREEN}./run_tests.sh dev${NC}  - Start LangGraph development server"
    ;;
esac

exit 0 