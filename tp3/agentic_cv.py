# First iteration
# Get a list of all people mentioned.
# For each person get the query that asks for the missing information to answer the question
# for each query bring context from pinecone
# Answer the question with the available context

# Second iteration
# Get a list of all people mentioned. Indicate if the query for person mentioned depends on information of other people {person: 'juan', depends_on:['Maria', 'Pedro']}
# For each person get the corresponding context
# for each query bring context from pinecone
# Answer the question with the available context
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Dict
from groq import Groq
import os
import copy
import streamlit as st
import pinecone as pc
from sentence_transformers import SentenceTransformer
from langgraph.graph import START
import re
import uuid
CV_DIRECTORY = "tp2/cvs"
INDEX_NAME = "cv-agent-index-v2"
client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

# --- Pinecone Integration ---
class PineconeRegistry:
    """Handles creation, population, and querying of a Pinecone index with namespaces."""
    _instance = None

    @staticmethod
    def get_instance(index_name: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
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
        """
        Populates the index with documents in a specific namespace.
        If the namespace does not exist, it is created.
        If the namespace exists, its content is overwritten.
        """
        if not documents:
            st.warning(f"No documents to populate for namespace: {namespace}")
            return

        # Check if namespace exists to provide informative output
        stats = self.pinecone_index.describe_index_stats()
        if namespace not in stats.namespaces:
            print(f"Namespace '{namespace}' does not exist. Creating and populating.")
        else:
            print(f"Namespace '{namespace}' already exists. Overwriting content.")
            # Clear existing vectors in the namespace before upserting new ones
            self.pinecone_index.delete(delete_all=True, namespace=namespace)

        vectors_to_upsert = []
        for i, doc in enumerate(documents):
            embedding = self.model.encode(doc).tolist()
            metadata = {"text": doc.strip()}
            vectors_to_upsert.append((f"doc-{i}", embedding, metadata))

        if vectors_to_upsert:
            self.pinecone_index.upsert(vectors=vectors_to_upsert, namespace=namespace)
            print(f"Indexed {len(vectors_to_upsert)} chunks for {namespace}.")

    def query(self, prompt: str, namespace: str, k: int = 4) -> list[str]:
        """Queries the index within a specific namespace."""
        query_vector = self.model.encode(prompt).tolist()
        matches = self.pinecone_index.query(
            vector=query_vector, top_k=k, include_metadata=True, namespace=namespace
        )[ "matches"]
        return [match["metadata"]["text"] for match in matches if "metadata" in match and "text" in match["metadata"]]



def call_model(prompt, system="You are a helpful assistant.") -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content or ""

# def call_model(messages: List[Dict[str, str]]) -> str:
#     chat_completion = client.chat.completions.create(
#         messages=messages,
#         model="llama-3.3-70b-versatile",
#     )
#     return chat_completion.choices[0].message.content or ""


class AgentState(TypedDict):
    query: str
    people_mentioned: Dict[str, List[str]]
    answer: str

def get_namespace_from_filename(filename):
    """Creates a clean namespace from the PDF filename."""
    name = os.path.splitext(filename)[0]
    # Sanitize the name to be a valid namespace
    return re.sub(r'[^a-zA-Z0-9_-]', '-', name)

def people_identifier(state: AgentState):
    allowed_people = []
    cv_files = [f for f in os.listdir(CV_DIRECTORY) if f.endswith(".pdf")]
    for filename in cv_files:
        pdf_path = os.path.join(CV_DIRECTORY, filename)
        allowed_people.append(get_namespace_from_filename(filename))
    allowed_people = ",".join(allowed_people)
    response = call_model(
        f"Your task is to return a list of People mentioned in this query. If the queries makes reference to the person writing the query you should include 'user' in the list. \
        The list should be separated by commas. \
        The list of people that can be referenced are: {allowed_people}\
        Present the people mentioend with the above format. \
        query:{state['query']}"
    )
    print(f"[people_identifier]: {response}")
    people_mentioned = response.split(",")
    people_dict = {}
    for person in people_mentioned:
        people_dict[person] = []
    return {'people_mentioned': people_dict}


def user_router(state: AgentState):
    pc_registry = PineconeRegistry.get_instance(INDEX_NAME)
    people_dict = copy.copy(state["people_mentioned"])
    for person in state["people_mentioned"]:
        query = call_model(
            f"For a given  text and for a given person. Formulate a question that will fetch the missing data? Only add the question and no other comments\
            For example: \
            Text: Who has more professional experience Anna or Peter? \
            Person: Anna \
            Question: How much professional experience does Anna have? \
            Text: {state['query']}\
            Person: {person}\
            Question:"
        )
        print(f"[user_route][{person}]: {query}")
        people_dict[person] = pc_registry.query(query, namespace = person.lower())
    return { 'people_mentioned': people_dict }

def aggregator(state: AgentState):
    context = [f"{person}: {' '.join(ctx)}" for person, ctx in state['people_mentioned'].items()]
    response = call_model(
        f"You're a human resources expert. Your job is to answer the questions about different candidates. For the given 'context' answer the 'question': \
        Context: {context} \
        Question: {state['query']}"
    )
    print(f"[aggregator]: {response}")
    return {"answer": response}

if __name__ == "__main__":
    builder = StateGraph(AgentState)
    builder.add_node("people_identifier", people_identifier)
    builder.add_node("user_router", user_router)
    builder.add_node("aggregator", aggregator)

    builder.add_edge(START, "people_identifier")
    builder.add_edge("people_identifier", "user_router")
    builder.add_edge("user_router", "aggregator")
    builder.add_edge("aggregator", END)

    query = "Cuantos años de experiencia tiene Anthony?"
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}
    state: AgentState =  {"query": query}

    graph.invoke(state, config)
    # while True:
    #     try:
    #         user_input = input("User: ")
    #         if user_input.lower() in ["quit", "exit", "q"]:
    #             print("Goodbye!")
    #             break
    #         graph.invoke(state)
    #     except:
    #         # fallback if input() is not available
    #         user_input = "What do you know about LangGraph?"
    #         print("User: " + user_input)
    #         stream_graph_updates(user_input)
    #         break
