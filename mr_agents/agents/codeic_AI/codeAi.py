import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# Ollama support via langchain_community
from langchain_community.llms import Ollama

# Gemini support
from langchain_google_genai import ChatGoogleGenerativeAI

print("Loading codeAi with Ollama and Gemini support...")
load_dotenv()


def create_code_agent():
    """
    Create a coding assistant agent that supports both Ollama and Gemini.
    Uses MODEL environment variable to select backend (default: ollama).
    """
    model = os.getenv("MODEL", "ollama").lower()
    
    if model == "ollama":
        # Use Ollama with default llama3 model
        llm = Ollama(model="llama3", base_url="http://localhost:11434")
        print("Using Ollama (llama3) as the coding assistant model")
    elif model.startswith("gemini"):
        # Use Google Generative AI (Gemini)
        google_key = os.getenv("GOOGLE_API_KEY")
        if not google_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is required in environment variables for Gemini access. "
                "Please set GOOGLE_API_KEY or use MODEL=ollama for Ollama backend."
            )
        os.environ["GOOGLE_API_KEY"] = google_key
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        print(f"Using Gemini ({model}) as the coding assistant model")
    else:
        raise ValueError(
            f"Unsupported MODEL: {model}. Use 'ollama' or 'gemini-*' variant."
        )
    
    return Agent(
        role="You are an expert code assistant. Write, review, debug, and refactor code with best practices.",
        goal="Provide high-quality code solutions with clear explanations and proper error handling.",
        backstory=(
            "You are a senior software engineer with expertise across multiple languages. "
            "You write clean, maintainable code following SOLID principles, provide detailed explanations, "
            "and always consider edge cases and error handling."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[],  # No web search tools like Serper
        llm=llm,
    )


def create_code_task(agent, prompt):
    """Create a coding task for the agent to execute."""
    return Task(
        description=f"Assist with coding task: {prompt}",
        expected_output=(
            "Well-structured code solution with explanations, error handling, "
            "relevant examples, and best practice recommendations."
        ),
        agent=agent,
    )


def run_code_agent(prompt):
    """Execute the coding assistant with the provided prompt."""
    agent = create_code_agent()
    task = create_code_task(agent, prompt)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print("CodeAI Result:", result)
    return result


if __name__ == "__main__":
    prompt = input("Enter your coding question or task: ").strip()
    if not prompt:
        prompt = "Write a Python function to validate email addresses with proper error handling."
    
    run_code_agent(prompt) 