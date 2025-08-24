import os
import re
import sys
import argparse
from typing import List, Dict, Any

# --- Conditional Import for Streamlit ---
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import pinecone as pc
from groq import Groq
from pypdf import PdfReader
from pypdf.errors import PdfStreamError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage

# --- Configuration ---
INDEX_NAME = "cv-agent-index-v3"
CANDIDATE_CVS = {
    "juan perez": "tp2/cvs/juan_perez.pdf",
    "maria garcia": "tp2/cvs/maria_garcia.pdf",
    "pedro rodriguez": "tp2/cvs/pedro_rodriguez.pdf",
    "leos": "tp2/cvs/leos_cv.pdf", # User's CV
    "scott sullivan": "tp2/cvs/scott_sullivan.pdf",
    "robert salazar": "tp2/cvs/robert_salazar.pdf",
    "anthony smith": "tp2/cvs/anthony_smith.pdf",
    "gerald jones": "tp2/cvs/gerald_jones.pdf",
    "ashley deleon": "tp2/cvs/ashley_deleon.pdf",
}
CANDIDATE_NAMES = list(CANDIDATE_CVS.keys())

# --- UI Abstraction Layer ---
class UIHandler:
    def __init__(self, is_gui: bool):
        self.is_gui = is_gui and STREAMLIT_AVAILABLE

    def spinner(self, text: str):
        if self.is_gui:
            return st.spinner(text)
        else:
            print(f"... {text}")
            class DummySpinner:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return DummySpinner()

    def success(self, text: str):
        if self.is_gui:
            st.sidebar.success(text)
        else:
            print(f"✅ {text}")

    def warning(self, text: str):
        if self.is_gui:
            st.warning(text)
        else:
            print(f"⚠️ {text}")

    def error(self, text: str):
        if self.is_gui:
            st.error(text)
        else:
            print(f"❌ {text}")

# --- State Management Abstraction ---
class SessionManager:
    def __init__(self, is_gui: bool):
        self.is_gui = is_gui and STREAMLIT_AVAILABLE
        if not self.is_gui:
            self._cli_state = {}

    def get(self, key: str, default: Any = None) -> Any:
        if self.is_gui:
            return st.session_state.get(key, default)
        else:
            return self._cli_state.get(key, default)

    def set(self, key: str, value: Any):
        if self.is_gui:
            st.session_state[key] = value
        else:
            self._cli_state[key] = value

# --- Helper Functions ---
def read_pdf(file_path: str, ui: UIHandler) -> str:
    try:
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text:
            ui.warning(f"No text extracted from {file_path}. The file might be empty or image-based.")
            return f"Este es el CV de {os.path.basename(file_path).split('.')[0]}. Actualmente esta vacio."
        return text
    except FileNotFoundError:
        ui.error(f"CV file not found at: {file_path}")
        return ""
    except PdfStreamError:
        ui.error(f"Could not read the PDF file at: {file_path}. It may be corrupted or not a valid PDF.")
        return ""
    except Exception as e:
        ui.error(f"An unexpected error occurred while reading {file_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 250) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return text_splitter.split_text(text)

# --- Pinecone Integration (Singleton) ---
class PineconeRegistry:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PineconeRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, index_name: str = INDEX_NAME, embedding_model_name: str = "all-MiniLM-L6-v2"):
        if not hasattr(self, 'initialized'):
            if not os.getenv("PINECONE_API_KEY"): raise ValueError("PINECONE_API_KEY not set.")
            pc_client = pc.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.model = SentenceTransformer(embedding_model_name)
            if index_name not in pc_client.list_indexes().names():
                print(f"Creating new Pinecone index: '{index_name}'...")
                dimension = self.model.get_sentence_embedding_dimension()
                pc_client.create_index(
                    name=index_name, dimension=dimension, metric="cosine",
                    spec=pc.ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Index '{index_name}' created.")
            self.pinecone_index = pc_client.Index(index_name)
            self.initialized = True

    def index_cv_if_needed(self, candidate_name: str, session: SessionManager, ui: UIHandler):
        indexed_cvs = session.get("indexed_cvs", set())
        if candidate_name not in indexed_cvs:
            with ui.spinner(f"Indexing {candidate_name.title()}'s CV..."):
                cv_path = CANDIDATE_CVS[candidate_name]
                cv_text = read_pdf(cv_path, ui)
                if cv_text:
                    documents = chunk_text(cv_text)
                    vectors = [(f"doc_{i}", self.model.encode(doc).tolist(), {"text": doc}) for i, doc in enumerate(documents)]
                    if vectors:
                        self.pinecone_index.delete(delete_all=True, namespace=candidate_name)
                        self.pinecone_index.upsert(vectors=vectors, namespace=candidate_name)
                        indexed_cvs.add(candidate_name)
                        session.set("indexed_cvs", indexed_cvs)
                        ui.success(f"Indexed {candidate_name.title()}")

    def query(self, prompt: str, namespace: str, k: int = 5) -> str:
        query_vector = self.model.encode(prompt).tolist()
        matches = self.pinecone_index.query(vector=query_vector, top_k=k, include_metadata=True, namespace=namespace)["matches"]
        return "\n---\n".join([m["metadata"]["text"] for m in matches])

# --- Specialist Agent and Orchestration ---
class CvAnalysisChatbot:
    def __init__(self, groq_client, is_gui: bool):
        self.groq_client = groq_client
        self.ui = UIHandler(is_gui)
        self.session = SessionManager(is_gui)
        self.pinecone_registry = PineconeRegistry()
        self._initialize_session_state()

    def _initialize_session_state(self):
        if self.session.get("messages") is None: self.session.set("messages", [])
        if self.session.get("specialist_agents") is None: self.session.set("specialist_agents", {})

    def get_or_create_specialist_agent(self, candidate_name: str):
        agents = self.session.get("specialist_agents", {})
        if candidate_name not in agents:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are an expert on the CV of {candidate_name.title()}. Answer questions based *only* on the provided CV context. Be concise. Respond in Spanish."),
                ("user", "Context:\n\n{context}\n\nQuestion: {question}")
            ])
            llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
            agents[candidate_name] = prompt | llm | StrOutputParser()
            self.session.set("specialist_agents", agents)
        return agents[candidate_name]

    def route_query(self, user_prompt: str) -> List[str]:
        if re.search(r'\b(yo|mi|tengo|mis)\b', user_prompt, re.IGNORECASE): return ["leos"]
        candidate_list = ", ".join(CANDIDATE_NAMES)
        prompt = f"From the text, identify which of these candidates are discussed: {candidate_list}. Respond with a comma-separated list of their names in lowercase. If none, respond 'None'.\n\nText: \"{user_prompt}\""
        response = self.groq_client.chat.completions.create(model="llama3-8b-8192", messages=[SystemMessage(content=prompt)], temperature=0)
        result = response.choices[0].message.content.lower()
        return [name.strip() for name in result.split(',') if name.strip() in CANDIDATE_NAMES]

    def synthesize_answers(self, question: str, individual_answers: dict) -> str:
        if not individual_answers: return "No information found."
        context = "\n\n".join([f"Analysis for {name.title()}:\n{answer}" for name, answer in individual_answers.items()])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an HR assistant. Synthesize the following analyses into a single, comparative answer to the user's original question. Respond in Spanish."),
            ("user", "Original Question: {question}\n\nIndividual Analyses:\n{context}\n\nSynthesized Answer:")
        ])
        llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})

    def get_response(self, prompt: str) -> str:
        """Processes a prompt and returns the agent's response."""
        with self.ui.spinner("Routing query and analyzing CVs..."):
            mentioned_candidates = self.route_query(prompt)
            if not mentioned_candidates:
                return "No specific candidate was identified. Please mention a name or use 'yo'/'mi'."
            elif len(mentioned_candidates) == 1:
                candidate = mentioned_candidates[0]
                self.pinecone_registry.index_cv_if_needed(candidate, self.session, self.ui)
                context = self.pinecone_registry.query(prompt, namespace=candidate)
                agent = self.get_or_create_specialist_agent(candidate)
                return agent.invoke({"context": context, "question": prompt})
            else:
                individual_answers = {}
                for candidate in mentioned_candidates:
                    self.pinecone_registry.index_cv_if_needed(candidate, self.session, self.ui)
                    context = self.pinecone_registry.query(prompt, namespace=candidate)
                    agent = self.get_or_create_specialist_agent(candidate)
                    answer = agent.invoke({"context": context, "question": prompt})
                    individual_answers[candidate] = answer
                return self.synthesize_answers(prompt, individual_answers)

def run_streamlit():
    """Main application loop for Streamlit GUI."""
    st.set_page_config(page_title="Per-Person HR Agent", page_icon="🧑‍💼")
    st.title("Per-Person HR Analyst Agent")
    st.sidebar.title("Available Candidates")
    st.sidebar.info("\n".join([f"- {name.title()}" for name in CANDIDATE_NAMES]))

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chatbot = CvAnalysisChatbot(groq_client=client, is_gui=True)

    messages = chatbot.session.get("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about one or more candidates..."):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            response = chatbot.get_response(prompt)
            st.markdown(response)
        messages.append({"role": "assistant", "content": response})
        chatbot.session.set("messages", messages)

def run_cli(question: str = None):
    """Main application loop for Command-Line Interface."""
    print("--- Per-Person HR Analyst CLI ---")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chatbot = CvAnalysisChatbot(groq_client=client, is_gui=False)

    if question:
        response = chatbot.get_response(question)
        print(f"\n🤖\n{response}")
        return

    print("Available candidates:", ", ".join(CANDIDATE_NAMES))
    print("Type 'quit' or 'exit' to end the session.")
    while True:
        try:
            prompt = input("\n> ")
            if prompt.lower() in ["quit", "exit"]:
                print("Session ended.")
                break
            response = chatbot.get_response(prompt)
            print(f"\n🤖\n{response}")
        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

def main():
    """Application entry point."""
    parser = argparse.ArgumentParser(description="CV Analysis Chatbot with multiple interfaces.")
    parser.add_argument("--no-gui", action="store_true", help="Run the application in command-line mode.")
    parser.add_argument("-q", "--question", type=str, help="Ask a single question and exit (requires --no-gui).")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("ERROR: GROQ_API_KEY and PINECONE_API_KEY must be set as environment variables.")
        sys.exit(1)

    if args.no_gui:
        run_cli(args.question)
    else:
        if args.question:
            print("ERROR: --question argument can only be used with --no-gui.")
            sys.exit(1)
        if not STREAMLIT_AVAILABLE:
            print("ERROR: Streamlit is not installed. Please install it (`pip install streamlit`) or run with --no-gui.")
            sys.exit(1)
        run_streamlit()

if __name__ == "__main__":
    main()
