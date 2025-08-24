import os
from typing import Dict, Type

import pinecone as pc
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.agents import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage

# --- Configuration ---
INDEX_NAME = "cv-agent-index-v2"
CANDIDATE_CVS = {
    "juan perez": "tp2/cvs/juan_perez.pdf",
    "maria garcia": "tp2/cvs/maria_garcia.pdf",
    "pedro rodriguez": "tp2/cvs/pedro_rodriguez.pdf",
    "leos": "tp2/cvs/leos_cv.pdf", # User's CV
}

# --- Helper Functions ---
def read_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text:
            st.warning(f"Warning: No text extracted from {file_path}. The file might be empty or image-based.")
            return f"Este es el CV de {os.path.basename(file_path).split('.')[0]}. Actualmente esta vacio."
        return text
    except FileNotFoundError:
        st.error(f"CV file not found at: {file_path}")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 250) -> list[str]:
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter.split_text(text)

# --- Pinecone Integration ---
class PineconeRegistry:
    """Handles creation, population, and querying of a Pinecone index with namespaces."""
    _instance = None

    @staticmethod
    def get_instance(index_name: str = INDEX_NAME, embedding_model_name: str = "all-MiniLM-L6-v2"):
        if PineconeRegistry._instance is None:
            PineconeRegistry._instance = PineconeRegistry(index_name, embedding_model_name)
        return PineconeRegistry._instance

    def __init__(self, index_name, embedding_model_name):
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
        
        pc_client = pc.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.model = SentenceTransformer(embedding_model_name)

        if index_name not in pc_client.list_indexes().names():
            st.info(f"Creating new Pinecone index: '{index_name}'...")
            dimension = self.model.get_sentence_embedding_dimension()
            pc_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=pc.ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            st.success(f"Index '{index_name}' created.")
        
        self.pinecone_index = pc_client.Index(index_name)

    def populate(self, documents: list[str], namespace: str):
        """Populates the index with documents in a specific namespace."""
        if not documents:
            st.warning(f"No documents to populate for namespace: {namespace}")
            return

        vectors_to_upsert = []
        for i, doc in enumerate(documents):
            embedding = self.model.encode(doc).tolist()
            metadata = {"text": doc.strip()}
            vectors_to_upsert.append((f"doc-{i}", embedding, metadata))

        if vectors_to_upsert:
            self.pinecone_index.delete(delete_all=True, namespace=namespace)
            self.pinecone_index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"Indexed {len(vectors_to_upsert)} chunks for {namespace}.")

    def query(self, prompt: str, namespace: str, k: int = 4) -> list[str]:
        """Queries the index within a specific namespace."""
        query_vector = self.model.encode(prompt).tolist()
        matches = self.pinecone_index.query(
            vector=query_vector, top_k=k, include_metadata=True, namespace=namespace
        )[ "matches"]
        return [match["metadata"]["text"] for match in matches if "metadata" in match and "text" in match["metadata"]]

# --- LangChain Agent Tool ---
pinecone_registry = PineconeRegistry.get_instance()

@tool
def get_cv_info(candidate_name: str, question: str) -> str:
    """
    Retrieves information about a specific candidate from their CV.
    Use this tool to answer any question about a candidate, including yourself.
    For questions about yourself (e.g., 'my experience', 'cuantos años tengo'), use 'leos' as the candidate_name.
    """
    normalized_name = candidate_name.lower()
    if normalized_name not in CANDIDATE_CVS:
        return f"Could not find a CV for {candidate_name}. Available candidates are: {', '.join(CANDIDATE_CVS.keys())}."

    # Check if this CV is already indexed in the current session
    if "indexed_cvs" not in st.session_state:
        st.session_state.indexed_cvs = set()

    if normalized_name not in st.session_state.indexed_cvs:
        with st.spinner(f"First-time setup for {normalized_name}'s CV..."):
            cv_path = CANDIDATE_CVS[normalized_name]
            cv_text = read_pdf(cv_path)
            if not cv_text:
                return f"Could not read the CV for {candidate_name}."
            
            documents = chunk_text(cv_text)
            pinecone_registry.populate(documents, namespace=normalized_name)
            st.session_state.indexed_cvs.add(normalized_name)
            st.sidebar.success(f"✅ Indexed {normalized_name.title()}")

    # Query for the specific information
    retrieved_docs = pinecone_registry.query(question, namespace=normalized_name)
    return "\n---\n".join(retrieved_docs) if retrieved_docs else "No relevant information found in the CV for that question."


# --- Streamlit Chat Application ---
class ChatApplication:
    def __init__(self, groq_client, tools: list[BaseTool]):
        self.agent_executor = self._create_agent_executor(groq_client, tools)
        self.initialize_session_state()

    def _create_agent_executor(self, groq_client, tools):
        """Creates the LangChain agent and executor."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful HR assistant. Your job is to answer questions about candidates based on their CVs.

- To answer, you MUST use the `get_cv_info` tool.
- For questions about the user (e.g., \"my experience\", \"cuantos años tengo\", \"mi cv\"), use 'leos' as the `candidate_name`.
- For comparison questions (e.g., \"who has more experience, Juan or Maria?\"\n), you must use the tool for EACH candidate separately and then combine their information to form your final answer.
- Always inform the user whose CV you are analyzing.
- Respond in Spanish."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def initialize_session_state(self):
        """Initializes Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def display_messages(self):
        """Displays chat messages from history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def run(self):
        """Main application loop."""
        st.title("HR Analyst Agent")
        st.sidebar.title("Available Candidates")
        st.sidebar.info("\n".join([f"- {name.title()}" for name in CANDIDATE_CVS.keys()]))

        self.display_messages()

        if prompt := st.chat_input("Pregunta sobre un candidato o compara varios..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analizando CVs..."):
                    result = self.agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    response = result.get("output", "No se pudo procesar la respuesta.")
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))


def main():
    """Application entry point."""
    st.set_page_config(page_title="HR Analyst Agent", page_icon="🤖")
    
    if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("GROQ_API_KEY and PINECONE_API_KEY must be set as environment variables.")
        st.stop()

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    tools = [get_cv_info]
    
    app = ChatApplication(groq_client=client, tools=tools)
    app.run()

if __name__ == "__main__":
    main()