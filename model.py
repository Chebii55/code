from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DB_FAISS_PATH = 'vectorstore/db_faiss'
MEMORY_DB_PATH = 'memory.db'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        session_id TEXT,
        user_input TEXT,
        bot_response TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_memory(session_id, user_input, bot_response):
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory (session_id, user_input, bot_response) VALUES (?, ?, ?)",
                   (session_id, user_input, bot_response))
    conn.commit()
    conn.close()

def retrieve_memory(session_id):
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_input, bot_response FROM memory WHERE session_id = ?", (session_id,))
    memory = cursor.fetchall()
    conn.close()
    return memory

def final_result(query, session_id):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    save_memory(session_id, query, response['result'])
    return response

@cl.on_chat_start
async def start():
    try:
        init_memory_db()
        session_id = str(cl.user_session.get("user_id"))
        chain = qa_bot()
        cl.user_session.set("chain", chain)
        cl.user_session.set("session_id", session_id)

        msg = cl.Message(content="Starting the oncologist bot...")
        await msg.send()
        msg.content = "Hi, Welcome to the Oncologist Bot. What is your query?"
        await msg.update()
    except Exception as e:
        logger.error(f"Error starting chat: {e}")

@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")
        session_id = cl.user_session.get("session_id")
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        # Retrieve previous memory
        memory = retrieve_memory(session_id)
        context = " ".join([f"User: {user_input} Bot: {bot_response}" for user_input, bot_response in memory])

        query = message.content
        response = await chain.acall({'query': query, 'context': context}, callbacks=[cb])
        answer = response["result"]
        sources = response["source_documents"]

        # Concise Response
        if sources:
            answer += "\nSources:\n" + "\n".join([f"- {doc.metadata['source']}" for doc in sources])
        else:
            answer += "\nNo sources found"

        await cl.Message(content=answer).send()
        save_memory(session_id, query, answer)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()
