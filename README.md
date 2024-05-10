# Advanced RAG Pipeline

The Advanced RAG Pipeline is a powerful system that leverages various state-of-the-art NLP models and techniques to perform semantic search and retrieval, reranking, and response generation. It is designed to work with pre-chunked documents from Hugging Face, generate embeddings using SentenceBERT, utilize FAISS for HNSW semantic search, rerank using distilBERT, and generate responses using OpenAI GPT3.5.

## Features

- Chunked Documents: The pipeline supports pre-chunked documents from Hugging Face, allowing for efficient processing and retrieval. These documents are a collection of research papers from arvix.

- Embedding Generation: SentenceBERT is used to generate high-quality embeddings for the documents, capturing their semantic meaning.

- Semantic Search and Retrieval: FAISS with HNSW index is employed for efficient semantic search and retrieval, enabling fast and accurate retrieval of relevant documents.

- Reranking: The pipeline utilizes distilBERT for reranking the retrieved documents, ensuring the most relevant documents are prioritized.

- Response Generation: OpenAI GPT3.5 is used to generate informative and contextually relevant responses based on the retrieved and reranked documents.

## Installation

To install and set up the Advanced RAG Pipeline, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/Kdotseth7/advanced-rag.git
    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3. Configure the pipeline:
    - Update the environment file `.env` with the following settings:

    ```shell
    OPENAI_API_KEY="your-openai-key"
    TOKENIZERS_PARALLELISM=true
    BATCH_SIZE=32
    MODEL_NAME="model-name"
    ```

4. Run the pipeline:

    ```shell
    python main.py
    ```


## Contributing

Contributions to the Advanced RAG Pipeline are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).