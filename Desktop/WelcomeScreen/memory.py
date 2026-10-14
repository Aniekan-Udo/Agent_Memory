import os

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver


driver = FalkorDriver(host="localhost", port=6379)

from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone
import json
from graphiti_core.llm_client.groq_client import GroqClient
from graphiti_core.llm_client.config import LLMConfig
from embedder import SentenceTransformerEmbedder

from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient


load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")

graphiti = Graphiti(
    graph_driver=driver,
    llm_client=GroqClient(config=LLMConfig(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        small_model="openai/gpt-oss-120b",  # or a smaller Groq model you trust
    )),
    embedder=SentenceTransformerEmbedder(),  # check its constructor args
    cross_encoder=BGERerankerClient()
)

async def write(user_id, graphiti):
    episodes = [
                {
                    "content": "Alex Chen joined the AI Research team in May 2022 as a Senior ML Engineer reporting to Maria Gomez.",
                    "type": EpisodeType.text,
                    "description": "employee spotlight",
                },
                {
                    "content": {
                        "name": "Maria Gomez",
                        "role": "Head of Product",
                        "start_date": "2019-03-01",
                        "reports_to": "CTO"
                    },
                    "type": EpisodeType.json,
                    "description": "HR profile export",
                },
            ]

    for i, ep in enumerate(episodes):
        await graphiti.add_episode(
            name=f"HR Episode {i}",
            episode_body=json.dumps(ep["content"]) if isinstance(ep["content"], dict) else ep["content"],
            source=ep["type"],
            source_description=ep["description"],
            reference_time=datetime.now(timezone.utc),
            group_id=user_id
        )


    return graphiti