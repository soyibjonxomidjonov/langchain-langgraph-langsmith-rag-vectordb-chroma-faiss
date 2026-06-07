from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from globals import gemini_api


# 1. Faylni yukla
loader = TextLoader("../test.txt", encoding="utf-8")
hujjatlar = loader.load()

# 2. Bo'laklarga ajrat
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # har bir bo'lak max 100 ta belgi
    chunk_overlap=20,  # bo'laklar orasida 20 ta belgi takrorlanadi
)
bolaklar = splitter.split_documents(hujjatlar)


# 3. Embedding modeli
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api
)


# 4. FAISS ga saqlash
print("Vector yaratilmoqda...")
db = FAISS.from_documents(bolaklar, embeddings)
print(f"Jami vectorlar: {db.index.ntotal}")

# 5. Qidirish
savol = "FastAPI nima?"
natijalar = db.similarity_search(savol, k=2)

# print(f"\nSavol: {savol}")
# print("--- Topilgan bo'laklar ---")
# for i, natija in enumerate(natijalar):
#     print(f"{i+1}. {natija.page_content}\n")

kontekst = "\n".join([n.page_content for n in natijalar])

print(kontekst)










