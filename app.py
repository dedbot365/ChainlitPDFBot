import os
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import chainlit as cl
from dotenv import load_dotenv

# Configure the API client
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    cl.user_session.set("messages", [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}])


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    context = "\n".join([doc.page_content for doc in docs])
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def save_user_info(name, phone, email):
    file_exists = os.path.isfile('user_info.csv')
    with open('user_info.csv', mode='a', newline='') as file:
        fieldnames = ['Name', 'Phone', 'Email']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': name, 'Phone': phone, 'Email': email})

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    pdf_text = get_pdf_text([file.path])
    text_chunks = get_text_chunks(pdf_text)
    get_vector_store(text_chunks)

    msg.content = f"Processing `{file.name}` done. You can now ask questions! Or you type 'call me' to get started."
    await msg.update()

    clear_chat_history()

@cl.on_message
async def on_message(message: cl.Message):
    collecting_info = cl.user_session.get("collecting_info", False)

    if collecting_info:
        # Collect user information
        if cl.user_session.get("name") is None:
            cl.user_session.set("name", message.content)
            await cl.Message(content="Please provide your phone number:").send()
        elif cl.user_session.get("phone") is None:
            cl.user_session.set("phone", message.content)
            await cl.Message(content="Please provide your email address:").send()
        elif cl.user_session.get("email") is None:
            cl.user_session.set("email", message.content)
            name = cl.user_session.get("name")
            phone = cl.user_session.get("phone")
            email = cl.user_session.get("email")
            save_user_info(name, phone, email)
            cl.user_session.set("collecting_info", False)
            await cl.Message(content=f"Thank you, {name}. We will contact you at {phone} or {email}.").send()
        return

    prompt = message.content
    if "call me" in prompt.lower():
        cl.user_session.set("collecting_info", True)
        cl.user_session.set("name", None)
        cl.user_session.set("phone", None)
        cl.user_session.set("email", None)
        await cl.Message(content="Please provide your name:").send()
        return

    response = user_input(prompt)
    await cl.Message(content=response).send()

    # Append user message to the chat history
    current_messages = cl.user_session.get("messages", [])
    current_messages.append({"role": "user", "content": prompt})
    cl.user_session.set("messages", current_messages)

