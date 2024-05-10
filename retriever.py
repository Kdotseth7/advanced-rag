from embeddings import Embeddings

class Retriever:
    @staticmethod
    def search(documents, embed_model, index, query, top_k) -> list:
        query_embedding = embed_model.get_embedding([query])
        distances, indices = index.search(query_embedding, top_k)
        return [documents[int(idx)]['text'] for idx in indices[0]]
