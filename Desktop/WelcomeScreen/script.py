import asyncio
import os
from datetime import datetime, timezone

from groq import Groq
from dotenv import load_dotenv

from memory import write, graphiti

load_dotenv()

GROQ_API_KEY= os.getenv("GROQ_API_KEY")

def llm(message:str):
    client = Groq(api_key=GROQ_API_KEY, base_url="https://api.groq.com")
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": message}],
        temperature=1,)
    return response.choices[0].message.content or " "



def context(results, llm, query):
    facts = "\n".join(f"- {r.fact}" for r in results)
    prompt = (
        "Use only the facts below as your memory. Do not invent anything; "
        "if the facts don't answer the question, say you don't know.\n\n"
        f"Facts:\n{facts}\n\nQuestion: {query}"
    )
    return llm(prompt)

#oday= datetime.now(timezone.utc).strftime("%A, %d %B %Y")
DISTILL_PROMPT = """You extract lasting facts about people and their habits for a memory system.

The shop owner's name is {name}. Today is {today}.
Message: "{message}"

Rewrite any lasting information as 1-3 short, plain sentences, using the exact names from the message.
Keep: who bought what and how much (items and quantities), nicknames and aliases
("Mama N is Mama Nkechi"), standing orders ("Mr James's usual is 100 bags of rice"),
preferences, and relationships between people.
Leave out: prices, totals, payments, balances, and debts.
Turn relative dates ("yesterday", "on Monday") into real dates using today's date.
Use only what the message says. Do not guess.
If there is nothing to keep, reply with exactly: NONE
Output only the sentences, or NONE."""

async def distill(user_name: str, message: str):
    prompt = DISTILL_PROMPT.format(name=user_name, message=message)
    result = (await asyncio.to_thread(llm, prompt)).strip()
    return None if result.upper().startswith("NONE") else result


async def remember(user_id: str, user_name: str, message: str):
    facts = await distill(user_name, message)
    if facts:
        await write(user_id, graphiti, facts)

async def main(user_id, user_name):
    query = "Where does kanny work, and what does he work as?."
    results = await graphiti.search(query, group_ids=[user_id])
    print(context(results, llm, query))
    await remember(user_id, user_name, query)

if __name__ == "__main__":
    asyncio.run(main("123", "Kanny"))





