from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from globals import groq_api


model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api)




# @tool
# def qoshish(a: int, b: int) -> int:
#     """Ikkita sonni qo'shadi."""
#     result = a - b
#     print(f"🔧 TOOL ISHLADI: qoshish({a}, {b}) = {result}")
#     return result
#
# @tool
# def kopaytirish(a: int, b: int) -> int:
#     """Ikkita sonni ko'paytiradi."""
#     result = a * b
#     print(f"🔧 TOOL ISHLADI: kopaytirish({a}, {b}) = {result}")
#     return result

@tool
def qoshish_keyin_kopaytir(a: int, b: int, c: int) -> int:
    """a va b ni qo'shadi, keyin natijani c ga ko'paytiradi."""
    qoshish_natija = a + b  # ← to'g'ri
    kopaytirish_natija = qoshish_natija * c
    return kopaytirish_natija

tools = [qoshish_keyin_kopaytir]


agent = create_agent(model,
                     tools,
                     system_prompt="Sen hech qachon o'zing hisoblama. "
           "Faqat berilgan toollardan foydalanib hisобла. "
           "Har bir qadam uchun tooldan kelgan natijani ishlat.")


javob = agent.invoke({
        "messages": [("human",
        "40 va 30 ni qoshish tool orqali qo'sh. "
        "Keyin faqat shu natijani 3 ga kopaytirish tool orqali ko'paytir. "
        "Har bir qadamni alohida bajар.")]
})

print("\nYakuniy javob:", javob["messages"][-1].content)


# barcha xabarlarni ko'r
# for xabar in javob["messages"]:
#     print(xabar)
#     print("---")
