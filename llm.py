from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(find_dotenv())

class LLM:
    """LLM Class for RAG Ppeline thats applied after Rrranking documents
    """
    def __init__(self, model: str, temperature: int) -> None:
        self.model = model
        self.temperature = temperature
        
    def generate(self, query: str, context: str) -> str:
        """Generate a response from OpenAI LLM using the given context
        """
        summary = """
        You're an assistant to anser questions using the given context.

        Context: {context}
        
        Answer the following question: {query}
        """
        llm = ChatOpenAI(temperature=self.temperature, model=self.model)
        prompt_template = PromptTemplate(input_variables=["context"], template=summary)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.invoke(input={"context": context, "query": query})
        return result["text"]
