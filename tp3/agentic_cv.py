import os
import re
import copy
import uuid
from typing import Dict, List, TypedDict

import pinecone as pc
import streamlit as st
from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CV_DIRECTORY = "tp2/cvs"
INDEX_NAME = "cv-agent-index-v2"

# --- API Clients ---
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()


# --- Pinecone Integration ---
class PineconeRegistry:
    """Handles creation, population, and querying of a Pinecone index with namespaces."""

    _instance = None

    @staticmethod
    def get_instance(index_name: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        if PineconeRegistry._instance is None:
            PineconeRegistry._instance = PineconeRegistry(
                index_name, embedding_model_name
            )
        return PineconeRegistry._instance

    def __init__(self, index_name, embedding_model_name):
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY environment variable is not set.")

        pc_client = pc.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.model = SentenceTransformer(embedding_model_name)

        if index_name not in pc_client.list_indexes().names():
            with st.spinner(f"Creating new Pinecone index: '{index_name}'..."):
                dimension = self.model.get_sentence_embedding_dimension()
                pc_client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=pc.ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            st.success(f"Index '{index_name}' created.")

        self.pinecone_index = pc_client.Index(index_name)

    def query(self, prompt: str, namespace: str, k: int = 4) -> list[str]:
        """Queries the index within a specific namespace."""
        query_vector = self.model.encode(prompt).tolist()
        matches = self.pinecone_index.query(
            vector=query_vector, top_k=k, include_metadata=True, namespace=namespace
        )["matches"]
        return [
            match["metadata"]["text"]
            for match in matches
            if "metadata" in match and "text" in match["metadata"]
        ]


# --- Agent Logic ---
def call_model(prompt, system="You are a helpful assistant.") -> str:
    """Function to call the Groq LLM."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return ""


class AgentState(TypedDict):
    query: str
    people_mentioned: Dict[str, List[str]]
    answer: str


def get_namespace_from_filename(filename):
    """Creates a clean namespace from the PDF filename."""
    name = os.path.splitext(filename)[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


# --- Agent Nodes ---
def people_identifier(state: AgentState):
    """Identifies people mentioned in the query from a list of available CVs."""
    st.write("Step 1: Identifying people mentioned in the query...")
    allowed_people = [
        get_namespace_from_filename(f)
        for f in os.listdir(CV_DIRECTORY)
        if f.endswith(".pdf")
    ]

    prompt = (
        f"Your task is to identify and list the people mentioned in the user's query. "
        f"The list of available people is: {', '.join(allowed_people)}. "
        f"Return only a comma-separated list of their names as they appear in the available list. Return the list only and no other comment."
        f"If just a first name is used and only one user has that name then you should use that one. e.g. Anthony -> anthony_smith"
        f"Query: {state['query']}"
    )
    response = call_model(prompt)
    print(allowed_people)
    print(response)
    people_mentioned = [
        p.strip() for p in response.split(",") if p.strip() in allowed_people
    ]
    people_dict = {person: [] for person in people_mentioned}

    st.write(
        f"Identified: `{', '.join(people_mentioned) if people_mentioned else 'None'}`"
    )
    return {"people_mentioned": people_dict}


def context_retriever(state: AgentState):
    """For each identified person, generates a query and retrieves context from their CV."""
    st.write("Step 2: Retrieving context for each person...")
    pc_registry = PineconeRegistry.get_instance(INDEX_NAME)
    people_dict = copy.copy(state["people_mentioned"])

    for person in state["people_mentioned"]:
        prompt = (
            f"I need to answer this question: '{state['query']}'. "
            f"What specific information should I look for in the CV of '{person}'? "
            f"Formulate a direct question to find that information. "
            f"For example, if the main query is 'Who has more experience?', your question for a person could be 'How much professional experience does this person have?'"
        )
        generated_query = call_model(prompt)

        with st.expander(f"Details for {person}"):
            st.write(f"Generated query: `{generated_query}`")
            context = pc_registry.query(generated_query, namespace=person.lower())
            people_dict[person] = context
            st.write("Retrieved context:")
            st.json(context if context else "No relevant context found.")

    return {"people_mentioned": people_dict}


def aggregator(state: AgentState):
    """Aggregates the retrieved context and generates a final answer."""
    st.write("Step 3: Generating the final answer...")
    context_str = "\n".join(
        [
            f"CV summary for {person}: ".join(ctx)
            for person, ctx in state["people_mentioned"].items()
        ]
    )

    prompt = (
        f"You are a helpful HR assistant."
        f"When evaluating professional experience. Examples:"
        f"Candidate 1: job A (2000 - 2015), job b (2010 - 2025). Candidate 1 the earliest year is 2000, the latest year is 2025. The candidate hasn't stopped working. So the answer is 2025 - 2000 = 25 years of experience"
        f"Candidate 2: job A (2000 - 2015), job b (2020 - 2025). Candidate 2 has worked from 2000 to 2015 and from 2020 to 2025 therefore he has (2015 - 2000) + (2025 - 2020) = 15 + 5 = 20 years of experience"
        f"Candidate 3: job A (2000 - 2018), job b (2010 - 2025), job c (2005 - 2024). Candidate 3 has worked from 2000 to 2025 uninterruptedly so he has 2025 - 2020 = 25 years of experience"
        f"Based on the following context from CVs, please answer the user's question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {state['query']}\n\n"
        f"Answer:"
    )

    response = call_model(prompt)
    st.write("Done!")
    return {"answer": response}


# --- Streamlit UI ---
def main():
    st.title("📄 Agentic CV Analyzer")
    st.markdown(
        "Ask a question about the candidates in the CV database. The agent will identify the relevant people, "
        "retrieve information from their CVs, and synthesize an answer."
    )

    # Check for API keys
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
        st.error(
            "Please set the GROQ_API_KEY and PINECONE_API_KEY environment variables."
        )
        st.stop()

    # Initialize the agent graph
    builder = StateGraph(AgentState)
    builder.add_node("people_identifier", people_identifier)
    builder.add_node("context_retriever", context_retriever)
    builder.add_node("aggregator", aggregator)

    builder.add_edge(START, "people_identifier")
    builder.add_edge("people_identifier", "context_retriever")
    builder.add_edge("context_retriever", "aggregator")
    builder.add_edge("aggregator", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # User input
    user_query = st.text_input(
        "Your question:", placeholder="e.g., Who has more experience, Anthony or Maria?"
    )

    if st.button("Analyze"):
        if user_query:
            with st.spinner("Agent is working..."):
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                initial_state = {"query": user_query}

                # The invoke method runs the graph and returns the final state
                final_state = graph.invoke(initial_state, config)

                st.divider()
                st.subheader("Answer")
                st.markdown(
                    final_state.get("answer", "Sorry, I couldn't find an answer.")
                )
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
