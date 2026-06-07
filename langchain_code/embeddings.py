# pip install langchain-huggingface sentence-transformers
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from globals import groq_api, gemini_api
#
#
#
# from langchain_huggingface import HuggingFaceEmbeddings
#
#
#
#
#
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=gemini_api
# )
#
#
#
# vector = embeddings.embed_query("FastAPI nima?")
#
# print(f"Vektor uzunligi: {len(vector)}")
# print(f"Birinchi 5 ta raqam: {vector[:5]}")



from langchain_google_genai import GoogleGenerativeAIEmbeddings
from globals import gemini_api

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api
)

vektor = embeddings.embed_query("FastAPI nima?")
print(f"Vektor uzunligi: {len(vektor)}")
print(f"Birinchi 5 ta raqam: {vektor[:5]}")


# import google.generativeai as genai
# from globals import gemini_api
#
# genai.configure(api_key=gemini_api)
#
# for m in genai.list_models():
#     if "embedContent" in m.supported_generation_methods:
#         print(m.name)