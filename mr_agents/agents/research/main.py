import os
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI

# Force cache invalidation - VERSION 2
print("Loading script version with Gemini support...")
load_dotenv()

serper_key = os.getenv("SERPER_API_KEY")
if serper_key:
    os.environ["SERPER_API_KEY"] = serper_key

search_tool = SerperDevTool()   

def create_research_agent(use_gpt=True):
    model = os.getenv("MODEL", "gemini-2.5-flash")
    
    if "gemini" in model.lower():
        # Use Google Gemini
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
    elif use_gpt or "gpt" in model.lower():
        # Use OpenAI GPT
        llm = ChatOpenAI(model=model, temperature=0.7)
    else:
        # Use Ollama for local models
        llm = Ollama(model=os.getenv("LLAMA2_MODEL", "llama2"), temperature=0.7)
    
    return Agent(
        role="You are a research assistant. Your task is to gather information on a given topic using the tools at your disposal. Use the web search tool to find relevant information and provide concise summaries of your findings.",
        goal="Gather information on a specific topic and provide a summary of your findings.",
        backstory="You are an expert researcher with access to powerful tools that allow you to quickly gather and summarize information from the web. Your goal is to assist users in finding accurate and relevant information on a wide range of topics.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )

def create_research_task(agent, topic):
    return Task(
        description=f"Research the topic: {topic}",
        expected_output="A concise, accurate research summary that answers user questions and lists top findings.",
        agent=agent
    )

def run_research(topic, use_gpt=True):
    agent = create_research_agent(use_gpt)
    task = create_research_task(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    print("Research Summary:", result)
if __name__ == "__main__":
    topic = "The impact of artificial intelligence on the job market"
    use_gpt = input("Use GPT-4 for research? (yes/no): ").strip().lower() == "yes" 
    topic = input("Enter the research topic: ").strip() or topic

    result = run_research(topic, use_gpt)
    print("Final Research Summary:", result)         