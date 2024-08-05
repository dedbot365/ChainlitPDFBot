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

# Load environment variables from a .env file and configure the Google API 
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """
    Retrieves the text content from a list of PDF documents.

    Args:
        pdf_docs (List[str]): A list of paths to the PDF documents.

    Returns:
        str: The concatenated text content from all the PDF documents.
    """
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the given text into chunks using the RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to be split.

    Returns:
        list: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    """
    Creates a vector store from the given chunks of text using GoogleGenerativeAIEmbeddings and FAISS.

    Args:
        chunks (List[str]): A list of text chunks from which to create the vector store.

    Returns:
        None: This function does not return anything. It saves the vector store locally as "faiss_index".
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Generates a conversational chain for answering questions.

    Returns:
        A conversational chain for answering questions.
    """
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
    """
    Clears the chat history by resetting the "messages" user session variable to a list containing a single dictionary with the role "assistant" and the content "Upload some PDFs and ask me a question".

    This function does not take any parameters.

    This function does not return anything.
    """
    cl.user_session.set("messages", [{"role": "assistant", "content": "Upload some PDFs and ask me a question"}])


def user_input(user_question):
    """
    Processes user input question through various steps involving embeddings, similarity search, and conversational chain to generate and return the output text response.
    
    Args:
        user_question (str): The user input question to be processed.
        
    Returns:
        str: The output text response generated based on the user input question.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    context = "\n".join([doc.page_content for doc in docs])
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def save_user_info(name, phone, email):
    """
    Saves user information to a CSV file.

    Args:
        name (str): The user's name.
        phone (str): The user's phone number.
        email (str): The user's email address.

    Returns:
        None
    """
    file_exists = os.path.isfile('user_info.csv')
    with open('user_info.csv', mode='a', newline='') as file:
        fieldnames = ['Name', 'Phone', 'Email']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': name, 'Phone': phone, 'Email': email})

@cl.on_chat_start
async def on_chat_start():
    """
    This function is an event handler for the `on_chat_start` event in the `chainlit` library. It is triggered when a chat session starts.

    The function prompts the user to upload a PDF file by sending a message with the content "Please upload a PDF file to begin!". It then waits for the user to upload a file until a file is successfully uploaded or the timeout of 180 seconds is reached.

    Once a file is uploaded, the function processes the file by extracting the text content using the `get_pdf_text` function. It then splits the text into smaller chunks using the `get_text_chunks` function. Finally, it creates a vector store using the `get_vector_store` function.

    After the file is processed, the function updates the message content to indicate that the processing is done and the user can now ask questions. It also clears the chat history.

    Parameters:
    None

    Returns:
    None
    """
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
    """
    This function handles incoming messages from the user.

    It checks if the application is currently collecting user information.
    If it is, the function processes the user's input to collect their name, phone number, and email address.
    Once all the information is collected, it saves the user's information and sends a confirmation message.

    If the application is not collecting user information, the function checks if the user's message contains the phrase "call me".
    If it does, the function starts the process of collecting user information.

    Otherwise, the function processes the user's message as a prompt, generates a response using the user_input function, and sends the response back to the user.
    The user's message is also appended to the chat history.

    Parameters:
    message (cl.Message): The incoming message from the user.

    Returns:
    None
    """
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

