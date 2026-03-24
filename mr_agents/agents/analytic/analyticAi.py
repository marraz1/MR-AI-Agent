import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

# optional search tool dependency; used only if SERPER_API_KEY is configured
try:
    from crewai_tools import SerperDevTool
except ImportError:
    SerperDevTool = None

print("Loading analyticAi with Gemini support...")
load_dotenv()


def create_analytic_agent():
    model = os.getenv("MODEL", "gemini-2.5-flash")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not google_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is required in environment variables for Gemini access."
        )

    # Set env var for library compatibility if needed
    os.environ["GOOGLE_API_KEY"] = google_key

    # Optional Serper web search tool
    tools = []
    serper_key = os.getenv("SERPER_API_KEY")
    if serper_key and SerperDevTool is not None:
        os.environ["SERPER_API_KEY"] = serper_key
        tools.append(SerperDevTool())

    llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)

    return Agent(
        role="You are a data analytics assistant. Gather insights and summarize results.",
        goal="Provide concise analytics insight, conclusions, and next steps.",
        backstory=(
            "You are an expert analytics agent that uses Gemini to analyze user prompts "
            "and deliver concise, factual summaries with recommendations."
        ),
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
    )


def create_analytic_task(agent, prompt):
    return Task(
        description=f"Analyze and summarize: {prompt}",
        expected_output=(
            "A concise analysis and actionable summary of the prompt, including findings "
            "and recommended next steps."
        ),
        agent=agent,
    )


def run_analytic(prompt):
    agent = create_analytic_agent()
    task = create_analytic_task(agent, prompt)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print("AnalyticAI Result:", result)
    return result


if __name__ == "__main__":
    prompt = input("Enter the analytics prompt: ").strip()
    if not prompt:
        prompt = "Analyze Q1 revenue trend and recommend optimizations."

    run_analytic(prompt)
