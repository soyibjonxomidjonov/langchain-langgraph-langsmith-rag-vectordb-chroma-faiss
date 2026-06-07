from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from globals import groq_api

# pip install langchain-community    # loader va splitter uchun
# from langchain_community.document_loaders import TextLoader        # loader







loader = TextLoader("../test.txt", encoding="utf-8")
hujjatlar = loader.load()
matn = hujjatlar[0].page_content


model = ChatGroq(
    model="llama-3.3-70b-versatile",
   temperature=0.3,
   api_key=groq_api
)
parser = StrOutputParser()
shablon = ChatPromptTemplate(
    [    ("system", "Sen yordamchi dasturchisin. Quyidagi ma'lumotlar asosida javob ber:\n\n{kontekst}"),
     MessagesPlaceholder(variable_name="tarix"),
     ("human", "{savol}")
     ]
)
zanjir = shablon | model | parser
tarix = []

print("Chatbot tayyor! (chiqish uchun 'exit' yoz)\n")


while True:
    savol = input("Savol: ")
    if savol.lower() == "exit":
        break

    javob = zanjir.invoke({
        "kontekst": matn,
        "tarix": tarix,
        "savol": savol
    })

    tarix.append(HumanMessage(content=savol))
    tarix.append(AIMessage(content=javob))

    print(f"Model: {javob}\n")






# print(type(hujjatlar))
# print(len(hujjatlar))
# print("---MATN---")
# print(hujjatlar[0].page_content)
# print("---METADATA---")
# print(hujjatlar[0].metadata)
