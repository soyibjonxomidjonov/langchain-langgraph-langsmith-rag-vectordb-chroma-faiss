from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from typing import TypedDict, List

from globals import groq_api


model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api)


class State(TypedDict):
    savol: str
    tahlil: str
    javob: str
    tekshiruv: str
    qayta_yoz: bool


# 1-node: savolni tahlil qiladi
def tahlil_node(state: State) -> State:
    print("📋 Tahlil node ishlayapti...")
    javob = model.invoke([HumanMessage(content=f"Bu savolni qisqacha tahlil qil: {state['savol']}")])

    return {
        "tahlil": javob.content
    }



# 2-node: javob yozadi
def javob_node(state: State) -> State:
    print("✍️  Javob node ishlayapti...")
    javob = model.invoke([
        HumanMessage(content=f" Savol: {state['savol']}\n"
                             f"Tahlil: {state['tahlil']}\n"
                     f"Endi to'liq javob yoz.")
    ])
    return {
        "javob": javob.content
    }


# 3-node: javobni tekshiradi
def tekshiruv_node(state: State) -> State:
    print("🔍 Tekshiruv node ishlayapti...")
    tekshiruv = model.invoke([
        HumanMessage(content=
                     f"Bu javob yaxshimi Faqat 'HA' yoki 'YOQ' deb javob ber:\n{state['javob']}")
    ])
    qayta = "YOQ" in tekshiruv.content.upper()
    print(f"   Tekshiruv natijasi: {'Qayta yoziladi' if qayta else 'Yaxshi!'}")
    return {
        "tekshiruv": tekshiruv.content,
        "qayta_yoz": qayta
    }

def qaror(state: State) -> State:
    if state['qayta_yoz']:
        return "javob"
    return END

# Graph yaratamiz
graph = StateGraph(State)
graph.add_node("tahlil", tahlil_node)
graph.add_node("javob", javob_node)
graph.add_node("tekshiruv", tekshiruv_node)




graph.set_entry_point("tahlil")
graph.add_edge("tahlil", "javob")
graph.add_edge("javob", "tekshiruv")




# Tekshiruvdan keyin qaror
graph.add_conditional_edges("tekshiruv", qaror)


app = graph.compile()
 
# Ishga tushurish
natija = app.invoke({
    "savol": "FastAPI nima va qachon yaratilgan?",
    "tahlil": "",
    "javob": "",
    "tekshiruv": "",
    "qayta_yoz": False
})


print("\n--- YAKUNIY JAVOB ---")
print(natija["javob"])













