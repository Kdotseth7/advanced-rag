from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())


class LLM:
    """LLM class for RAG pipeline after re-ranking documents."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def generate(self, query: str, context: str) -> str:
        """Generate a response from OpenAI LLM using the given context."""

        template = """
You are an assistant that answers questions using only the given context.

Context:
{context}

Question:
{query}

Answer:
"""

        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature
        )

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=template
        )

        chain = prompt | llm | StrOutputParser()

        result = chain.invoke({
            "context": context,
            "query": query
        })

        return result