import os
from uuid import uuid4

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

async def write(user_id, graphiti, content, reference_time=None):
    await graphiti.add_episode(
        name=f"chat-{user_id}-{uuid4().hex[:8]}",
        episode_body=content,
        source=EpisodeType.text,
        source_description="user chat message",
        reference_time=reference_time or datetime.now(timezone.utc),
        group_id=user_id,
    )