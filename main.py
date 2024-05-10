import os
import faiss
import numpy as np
from utils import Utils
from dataset import Dataset
from tqdm import tqdm
from embeddings import Embeddings
from retriever import Retriever
from reranker import Reranker
from llm import LLM


if __name__ == '__main__':
    # Load Utils
    utils = Utils()
    utils.check_dir(".indices")
    
    # Load the dataset (train split) already chunked
    dataset = Dataset("jamescalam/ai-arxiv-chunked", "train")
    documents = dataset.get_dataset()
        
    # Generate embeddings for the documents using SentenceBERT and index them using FAISS
    index = faiss.IndexFlatL2(768)
    sentence_bert = Embeddings("all-mpnet-base-v2")
    batch_size = int(os.getenv("BATCH_SIZE"))
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding Documents", colour="green"):
        batch = documents[i:i+batch_size]
        embeds = sentence_bert.get_embedding(batch["text"])
        # to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
        # index.add(np.array(to_upsert))
        index.add(np.array(embeds))
        
    # Save the index
    faiss.write_index(index, ".indices/index_latest.idx")
    
    # Retrieve the top-k documents for a query using the FAISS index
    query = "Can you explain why we would want to do RLHF?"
    retriever = Retriever()
    docs = retriever.search(documents=documents, embed_model=sentence_bert, index=index, query=query, top_k=20)
    
    # Rerank the top-k documents using DistilBERT
    reranker = Reranker("sentence-transformers/msmarco-distilbert-base-v3")
    reranked_docs = reranker.rerank(docs, query, top_k=5)
    context = "\n".join([doc[0] for doc in reranked_docs])
    
    # Generate response from OPENAI Model
    llm = LLM(model="gpt-3.5-turbo", temperature=0)
    llm_response = llm.generate(query=query, context=context)
    print(llm_response)