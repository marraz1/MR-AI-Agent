import os
from dotenv import load_dotenv

load_dotenv()

print("MODEL env var:", repr(os.getenv("MODEL")))
print("MODEL with default:", repr(os.getenv("MODEL", "gpt-4")))

# Test the exact line from the error
model = os.getenv("MODEL", "gpt-4")
print("model variable:", repr(model))

# Test ChatOpenAI import
try:
    from langchain_openai import ChatOpenAI
    print("ChatOpenAI import successful")
    # Try to create with the model
    llm = ChatOpenAI(model=model, temperature=0.7)
    print("ChatOpenAI creation successful")
except Exception as e:
    print("Error:", e)