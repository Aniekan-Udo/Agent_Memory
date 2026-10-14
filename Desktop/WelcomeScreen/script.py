import asyncio
import os

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



async def main(user_id, graphiti):
    query = ("Who is head of product?")
    await graphiti.build_indices_and_constraints()
    await write(user_id, graphiti)
    results = await graphiti.search(query, group_ids=[user_id])
    print(context(results, llm, query))




if __name__ == "__main__":
    asyncio.run(main(user_id="123", graphiti=graphiti))



