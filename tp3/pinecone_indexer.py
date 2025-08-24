import os
import re
from pypdf import PdfReader
from agentic_cv import PineconeRegistry  # Import the custom class

# --- Configuration ---
CV_DIRECTORY = "tp2/cvs"
INDEX_NAME = "cv-agent-index-v2"  # Match the index name from agentic-cv

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def get_namespace_from_filename(filename):
    """Creates a clean namespace from the PDF filename."""
    name = os.path.splitext(filename)[0]
    # Sanitize the name to be a valid namespace
    return re.sub(r'[^a-zA-Z0-9_-]', '-', name)

def main():
    """
    Indexes CVs from a directory into a Pinecone index with namespaces
    using the PineconeRegistry class.
    """
    # --- 1. Get PineconeRegistry Instance ---
    # The get_instance method handles initialization of Pinecone and the model.
    # Make sure PINECONE_API_KEY environment variable is set.
    try:
        print("Initializing PineconeRegistry...")
        pc_registry = PineconeRegistry.get_instance(INDEX_NAME)
        print("PineconeRegistry initialized successfully.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the PINECONE_API_KEY environment variable.")
        return

    # --- 2. Process and Populate CVs ---
    print(f"Processing CVs from '{CV_DIRECTORY}'...")
    cv_files = [f for f in os.listdir(CV_DIRECTORY) if f.endswith(".pdf")]

    if not cv_files:
        print(f"No PDF files found in '{CV_DIRECTORY}'.")
        return

    for filename in cv_files:
        pdf_path = os.path.join(CV_DIRECTORY, filename)
        namespace = get_namespace_from_filename(filename)

        print(f"\nProcessing {filename} for namespace '{namespace}'...")

        # Extract text
        cv_text = extract_text_from_pdf(pdf_path)
        if not cv_text:
            print(f"Skipping {filename} due to extraction error or empty content.")
            continue

        # The populate method expects a list of documents (chunks).
        # For this example, we'll treat the entire CV text as a single document.
        documents = [cv_text]

        # Populate the index for the given namespace
        pc_registry.populate(documents=documents, namespace=namespace)
        print(f"Finished processing for namespace '{namespace}'.")

    print("\nIndexing complete.")
    print(f"You can now query the '{INDEX_NAME}' index using the namespaces.")

if __name__ == "__main__":
    # Note: The PineconeRegistry class uses streamlit for info messages,
    # which will print to the console when run as a script.
    main()
