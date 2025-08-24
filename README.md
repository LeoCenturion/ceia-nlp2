# NLP2 Project

This project contains a collection of scripts and applications for Natural Language Processing tasks, including a CV generation utility, a chatbot for querying CVs, and an agentic CV analyzer.

## Prerequisites

- Python 3.13 or higher
- [Poetry](https://python-poetry.org/docs/) for dependency management

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd nlp2
    ```

2.  **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

3.  **Activate the virtual environment:**

    ```bash
    poetry env activate
    ```

4.  **API Keys:**

    This project requires API keys for Groq and Pinecone. You will need to set these as environment variables or place them directly in the scripts that require them (not recommended for production).

    -   `GROQ_API_KEY`: Your API key for the Groq LLM service.
    -   `PINECONE_API_KEY`: Your API key for the Pinecone vector database.

## Usage

### 1. Generate Sample CVs

The `generate_cvs.py` script creates five random CVs in PDF format and saves them to the `tp2/cvs/` directory. These CVs are used as sample data for the chatbot.

To run the script:

```bash
python generate_cvs.py
```

### 2. Run the CV Chatbot

The `tp2/chatbot.py` script launches a Streamlit web application that allows you to chat with a bot to find information within the generated CVs.

To run the chatbot:

```bash
streamlit run tp2/chatbot.py
```

### 3. Agentic CV Analyzer

The `tp3/agentic-cv.py` script is an agent-based system for analyzing CVs using `langgraph`. This is a more advanced script and may require further configuration.

To run the analyzer:

```bash
python tp3/agentic-cv.py
```

## Project Structure

-   `pyproject.toml`: Project configuration and dependencies for Poetry.
-   `tp1/`: Contains notebooks and scripts related to TinyGPT.
-   `tp2/`: Contains the CV chatbot application and related files.
-   `tp3/`: Contains the agentic CV analyzer.
