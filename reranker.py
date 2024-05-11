from sentence_transformers import SentenceTransformer, util


class Reranker:
    """
    Reranks the top-k documents retrieved by the retriever using a SentenceTransformer model.
    """
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(self, documents, query, top_n=10) -> list:
        model = SentenceTransformer(self.model_name)
        query_embedding = model.encode(query, convert_to_tensor=True)
        document_embeddings = model.encode(documents, convert_to_tensor=True)
        
        # Compute cosine similarities between the query and all documents
        similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

        # Combine the scores with the documents
        doc_scores = zip(documents, similarities)

        # Sort documents by their scores in descending order
        reranked_documents = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return reranked_documents[:top_n]