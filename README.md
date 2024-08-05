# Project README

## Overview

This project demonstrates a sophisticated PDF processing and question-answering system using various Python libraries and frameworks. The system allows users to upload PDF documents, processes the text content, and enables users to ask questions based on the content of the uploaded documents. Additionally, it includes functionality for collecting user information through an interactive chat interface.

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









