import os
from dotenv import load_dotenv
load_dotenv()


openai_api = os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

gemini_api = os.environ["GEMiNI_API_KEY"] = os.environ.get("GEMiNI_API_KEY")

groq_api = os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")



langsmith_api = os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
# LangSmith uchun shu 3 ta qator kerak
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api
os.environ["LANGCHAIN_PROJECT"] = "langchain-darslari"