import argparse
import os
import sys
from typing import Dict

import pinecone as pc
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



def read_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def read_text_file(file_path: str) -> str:
    """Reads content from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 150) -> list[str]:
    """Splits text into chunks using a recursive character splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


index_name = "personal-cv-index"

class PineconeRegistry:
    """Handles querying the Pinecone vector database."""

    @staticmethod
    def create(index_name: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Creates a PineconeRegistry instance, creating the index if it doesn't exist."""
        pc_client = pc.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        model = SentenceTransformer(embedding_model_name)

        if index_name not in pc_client.list_indexes().names():
            st.write(f"Creating index '{index_name}'...")
            dimension = model.get_sentence_embedding_dimension()
            pc_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=pc.ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            st.write("Index created.")

        index = pc_client.Index(index_name)
        return PineconeRegistry(pinecone_index=index, embedding_model=model)

    def __init__(self, pinecone_index, embedding_model):
        self.pinecone_index = pinecone_index
        self.embedding_model = embedding_model

    def populate(self, documents: list[str]):
        """Populates the Pinecone index with documents, storing text in metadata."""
        vectors_to_upsert = []
        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode(doc).tolist()
            doc_id = f"cv-{i}"
            # Store the document text in metadata
            metadata = {"text": doc.strip()}
            vectors_to_upsert.append((doc_id, embedding, metadata))

        # Upsert in batches to avoid overwhelming the service
        if vectors_to_upsert:
            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i : i + 100]
                self.pinecone_index.upsert(vectors=batch)

    def query(self, prompt: str, k: int = 3) -> list[str]:
        """Queries the index and retrieves document text from metadata."""
        query_vector = self.embedding_model.encode(prompt).tolist()
        matches = self.pinecone_index.query(
            vector=query_vector, top_k=k, include_metadata=True
        )["matches"]

        docs = []
        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                docs.append(match["metadata"]["text"])
        return docs
    # 
    # 
class ContextProvider:
    def expand_context(self, user_prompt: str) -> str:
        """Adds context to a user prompt."""
        raise NotImplementedError


class CvContextProvider(ContextProvider):
    """Collaborates with PineconeRegistry to format context for the prompt."""

    def __init__(self, registry: PineconeRegistry):
        self.registry = registry

    def expand_context(self, user_prompt: str) -> str:
        """Expands the user prompt with context from Pinecone."""
        context_docs = self.registry.query(user_prompt)
        context = ",".join(doc.strip() for doc in context_docs)
        return f"Usa los siguientes CVs {context} para resolver {user_prompt}"

class PersonalCvContextProvider(ContextProvider):
    """Collaborates with PineconeRegistry to format context for the prompt."""

    def __init__(self, registry: PineconeRegistry):
        self.registry = registry

    def expand_context(self, user_prompt: str) -> str:
        """Expands the user prompt with context from Pinecone."""
        context_docs = self.registry.query(user_prompt)
        context = ",".join(doc.strip() for doc in context_docs)
        return f"Usa los siguientes fragmentos: ```{context}``` para resolver {user_prompt}"

class ChatApplication:
    def __init__(self, groq_client, context_provider: ContextProvider, system_prompt: str):
        self.groq_client = groq_client
        self.context_provider = context_provider
        self.system_prompt = system_prompt
        self.initialize_session_state()

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []


    def display_messages(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def run(self):
        st.title("Streamlit Chat with Groq Llama")
        self.display_messages()

        st.sidebar.title("⚙️ Configuración del Chatbot")
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎭 Personalidad del Bot")

        system_prompt = st.sidebar.text_area(
            "Mensaje del sistema:",
            value=self.system_prompt,
            height=100,
            help="Dile a la IA cuales caracteristicas quieres en tu proximo empleado.",
        )

        st.sidebar.subheader("🧠 Configuración de Memoria")
        conversational_memory_length = st.sidebar.slider(
            "Longitud de la memoria conversacional:",
            min_value=1,
            max_value=10,
            value=5,
            help="Número de intercambios anteriores que el bot recordará. Más memoria = mayor contexto pero mayor costo computacional",
        )

        if st.sidebar.button("🗑️ Limpiar Conversación"):
            st.session_state.messages = []
            st.sidebar.success("✅ Conversación limpiada")
            st.rerun()

        # React to user input
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            user_input = self.context_provider.expand_context(prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                messages_for_api = [{"role": "system", "content": system_prompt}]

                # Sliced history
                history_messages = st.session_state.messages[:-1]
                sliced_history = history_messages[-(conversational_memory_length * 2) :]
                messages_for_api.extend(sliced_history)

                messages_for_api.append({"role": "user", "content": user_input})

                stream = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages_for_api,
                    stream=True,
                )
                response = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    parser = argparse.ArgumentParser(description="Run Streamlit chatbot or index documents.")
    parser.add_argument(
        "--index",
        metavar="FILE_PATH",
        help="Path to a PDF or text file to index into Pinecone. If not provided, runs the Streamlit app.",
        required = False

    )
    args = parser.parse_args()

    if args.index:
        file_path = args.index
        pinecone_registry = PineconeRegistry.create(index_name=index_name)
        documents = []

        if file_path.endswith(".pdf"):
            print(f"Reading PDF file: {file_path}")
            content = read_pdf(file_path)
        else:
            print(f"Reading text file: {file_path}")
            content = read_text_file(file_path)

        documents = chunk_text(content)
        print(f"Chunked {len(documents)} documents.")
        print("Populating Pinecone index...")
        pinecone_registry.populate(documents)
        print("Indexing complete!")
    else:
        # Streamlit app mode
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        pinecone_registry = PineconeRegistry.create(index_name=index_name)
        context_provider = PersonalCvContextProvider(registry=pinecone_registry)
        system_prompt = "Eres un asistente experto en analizar CVs. Responde preguntas basándote en el contenido del CV proporcionado. Céntrate únicamente en la última pregunta del usuario; el resto de la conversación es para darte contexto."
        app = ChatApplication(
            groq_client=client,
            context_provider=context_provider,
            system_prompt=system_prompt,
        )
        app.run()

if __name__ == "__main__":
    main()
