Project README
Overview
This project demonstrates a sophisticated PDF processing and question-answering system using various Python libraries and frameworks. The system allows users to upload PDF documents, processes the text content, and enables users to ask questions based on the content of the uploaded documents. Additionally, it includes functionality for collecting user information through an interactive chat interface.

Key Components
PDF Processing: Extracts text from PDF files.
Text Chunking: Splits large text into manageable chunks.
Embeddings and Vector Store: Uses Google's Generative AI for embeddings and FAISS for vector storage.
Conversational Chain: Handles the QA process based on the provided context.
User Interaction: Manages user input and stores user information.
Setup
Prerequisites
Python 3.7 or higher
Virtual environment (recommended)
API key for Google Generative AI
Installation
Clone the repository:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Configure environment variables:
Create a .env file in the root directory of the project and add your Google API key:

env
Copy code
GOOGLE_API_KEY=your_google_api_key
Usage
Running the Application
Start the application:

bash
Copy code
chainlit run main.py
Upload a PDF file:
When prompted, upload a PDF file to begin the processing.

Ask Questions:
Once the PDF processing is complete, you can ask questions based on the content of the uploaded document.

Collect User Information:
You can also trigger a sequence to collect user information by typing "call me".

Functions
get_pdf_text(pdf_docs):
Extracts text from a list of PDF files.

get_text_chunks(text):
Splits the text into chunks using RecursiveCharacterTextSplitter.

get_vector_store(chunks):
Generates embeddings and stores them in a FAISS vector store.

get_conversational_chain():
Sets up the conversational chain for QA.

clear_chat_history():
Clears the chat history for a new session.

user_input(user_question):
Processes the user question and retrieves an answer from the context.

save_user_info(name, phone, email):
Saves user information to a CSV file.

Event Handlers
on_chat_start():
Handles the initial chat setup and PDF upload.

on_message(message):
Processes user messages and handles different interactions including collecting user information.

Contributing
Feel free to submit issues or pull requests if you have any improvements or bug fixes.

License
This project is licensed under the MIT License.

Acknowledgments
LangChain
Google Generative AI
FAISS
Chainlit
Note: This project assumes that you have a basic understanding of Python and virtual environments. For detailed usage of the libraries, please refer to their official documentation.
