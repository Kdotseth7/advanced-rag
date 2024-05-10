from sentence_transformers import SentenceTransformer


class Embeddings:
    """
    Generates embeddings for a given text using a SentenceBERT model.
    """
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_embedding(self, text) -> list:
        model = SentenceTransformer(self.model_name)  # all-mpnet-base-v2
        embeddings = model.encode(text)
        return embeddings
