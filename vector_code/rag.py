import os

from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from globals import gemini_api, groq_api

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api
)

if os.path.exists("faiss_index"):
    print("FAISS diskdan yuklanmoqda...")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

else:
    print("FAISS yangi yaratilmoqda...")
    loader = TextLoader("../test.txt", encoding="utf-8")
    hujjatlar = loader.load()
    # 2. Bo'laklarga ajrat
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20,
    )
    bolaklar = splitter.split_documents(hujjatlar)
    db = FAISS.from_documents(bolaklar, embeddings)
    db.save_local("faiss_index")
    print("FAISS saqlandi!")

tarix = []


model = ChatGroq(
    model="llama-3.3-70b-versatile",
   temperature=0.3,
   api_key=groq_api
)
parser = StrOutputParser()


shablon = ChatPromptTemplate.from_messages([
    ("system",
     "Sen faqat quyidagi ma'lumotlar asosida javob berasan.\n"
     "Agar javob ma'lumotlarda bo'lmasa 'Bilmayman' de.\n"
     "O'z bilimingdan HECH QACHON foydalanma.\n"
     "Ma'lumotlarda noto'g'ri bo'lsa ham, shuni ayt.\n\n"
     "Ma'lumotlar:\n{kontekst}"),
    MessagesPlaceholder(variable_name="tarix"),
    ("human", "{savol}")
])


zanjir = shablon | model | parser

print("RAG tayyor!\n")
while True:
    savol = input("Savol: ")
    if savol.lower() == "exit":
        break

    natijalar = db.similarity_search(savol, k=2)
    kontekst = "\n".join([n.page_content for n in natijalar])
    javob = zanjir.invoke({
        "tarix": tarix,
        "kontekst": kontekst,
        "savol": savol
    })

    tarix.append(HumanMessage(content=savol))
    tarix.append(AIMessage(content=javob))
    print(f"Model: {javob}\n")














