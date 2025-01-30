from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def get_chain(model_name, temperature=0.7):
    """
    Creates a chain with the specified model and temperature.
    """
    # Initialize ChatGroq
    lama = ChatGroq(
        temperature=temperature,
        groq_api_key=groq_api_key,
        model_name=model_name,
    )

    # Output parser
    parser = StrOutputParser()

    # ChatPromptTemplate
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I need more context".

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define chain
    chain = prompt | lama | parser
    return chain

def ask_question(chain, question, context, conversation_history=""):
    """
    Generate a response to the user's question using the provided chain.
    """
    full_context = f"{conversation_history}\nContext: {context}\nQuestion: {question}"
    formatted_input = {"context": full_context, "question": question}
    response = chain.invoke(formatted_input)
    conversation_history += f"\nQuestion: {question}\nAnswer: {response}"
    return response, conversation_history