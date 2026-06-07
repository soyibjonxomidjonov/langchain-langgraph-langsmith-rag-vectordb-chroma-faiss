from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_groq import ChatGroq

from globals import groq_api, langsmith_api

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api)

# 1. State — qadamlar orasida ma'lumot saqlanadi
class State(TypedDict):
    son_a: int
    son_b: int
    qoshish_natija: int
    kopaytirish_natija: int



# 2. Nodelar — har bir qadam uchun funksiya
def qoshish_node(state: State) -> State:
    natija = state["son_a"] +state["son_b"]
    print(f"Qoshish node: {state['son_a']} + {state['son_b']} = {natija}")
    return {"qoshish_natija": natija}

def kopaytirish_node(state: State) -> State:
    natija = state["qoshish_natija"] * 3
    print(f"Kopaytirish node: {state['qoshish_natija']} * 3 = {natija}")
    return {"kopaytirish_natija": natija}


graph = StateGraph(State)

# Tartib: qoshish → kopaytirish → END
graph.add_node("qoshish", qoshish_node)
graph.add_node("kopaytirish", kopaytirish_node)






# Tartib: qoshish → kopaytirish → END
graph.set_entry_point("qoshish")
graph.add_edge("qoshish", "kopaytirish")
graph.add_edge("kopaytirish", END)


app = graph.compile()






# 4. Ishga tushir
natija = app.invoke(
    {
        "son_a": 40,
        "son_b": 30,
        "qoshish_natija": 0,  # boshlang'ich qiymat
        "kopaytirish_natija": 0  # boshlang'ich qiymat
    }
)

print(f"\nYakuniy natija: {natija['kopaytirish_natija']}")


































