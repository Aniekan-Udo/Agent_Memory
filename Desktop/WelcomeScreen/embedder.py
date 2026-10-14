import asyncio

from sentence_transformers import SentenceTransformer
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig


class SentenceTransformerEmbedder(EmbedderClient):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.config = EmbedderConfig(embedding_dim=1024)

    async def create(self, input_data):
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = list(input_data)
        vectors = await asyncio.to_thread(self.model.encode, texts)
        return vectors[0].tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(self.model.encode, input_data_list)
        return vectors.tolist()
