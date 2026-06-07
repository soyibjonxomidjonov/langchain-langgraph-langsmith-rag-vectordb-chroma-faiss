from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


from globals import groq_api

model = ChatGroq(
    model="llama-3.3-70b-versatile",
   temperature=0.3,
   api_key=groq_api
)
parser = StrOutputParser()

shablon = ChatPromptTemplate(
    [("system", "Sen aql va foydali chatbotsan. Foydalanuvchining savollariga aniq va tushunarli javoblar berishing kerak."
                "Agar javobni bilmasang,"
                "'Bilmadim' deb javob berishing kerak."
                "Va sen odam bilan to'liq ravishda suhbatlashib uni dardlarini va qayg'ulariga sherik bo'lishing kerak"),
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
        "tarix": tarix,
        "savol": savol
    })

    tarix.append(HumanMessage(content=savol))
    tarix.append(AIMessage(content=javob))

    print(f"Model: {javob}\n")



















