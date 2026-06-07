from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# pip install langchain-text-splitters  # RecursiveCharacterTextSplitter uchun
# from langchain_text_splitters import RecursiveCharacterTextSplitter # splitter



loader = TextLoader("../test.txt", encoding="utf-8")
hujjatlar = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, # har bir bo'lak max 100 ta belgi
    chunk_overlap=20,  # bo'laklar orasida 20 ta belgi takrorlanadi
)

bolaklab = splitter.split_documents(hujjatlar)


print(f"Jami bo'laklar soni: {len(bolaklab)}\n")


for i, bolak  in enumerate(bolaklab):
    print(f"---{i+1} -bo'lak---")
    print(bolak.page_content)
    print(f"Belgilar soni: {len(bolak.page_content)}\n")
