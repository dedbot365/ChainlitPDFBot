# Project README

## Overview

This project demonstrates a sophisticated PDF processing and question-answering system using various Python libraries and frameworks. The system allows users to upload PDF documents, processes the text content, and enables users to ask questions based on the content of the uploaded documents. Additionally, it includes functionality for collecting user information through an interactive chat interface using Chainlit.

## Key Components

- **PDF Processing**: Extracts text from PDF files.
- **Text Chunking**: Splits large text into manageable chunks.
- **Embeddings and Vector Store**: Uses Google's Generative AI for embeddings and FAISS for vector storage.
- **Conversational Chain**: Handles the QA process based on the provided context.
- **User Interaction**: Manages user input and stores user information.

## Setup

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- API key for Google Generative AI

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`       
3. **Install the required packages**:
     ```bash
      pip install -r requirements.txt  #Use this to install the required packages
4. **Configure environment variables**:
   Create a .env file in the root directory of the project and add your Google API key:

   ```bash
      GOOGLE_API_KEY=your_google_api_key

## Usage
### Running the Application
1. **Start the application**:
     ```bash
      chainlit run main.py
2. **Upload a PDF file**:
   When prompted, upload a PDF file to begin the processing.

3. **Ask Questions**:
   Once the PDF processing is complete, you can ask questions based on the content of the uploaded document.

4. **Collect User Information**:
   You can also trigger a sequence to collect user information by typing "call me".
   
## Note
This project assumes that you have a basic understanding of Python and virtual environments. For detailed usage of the libraries, please refer to their official documentation.











