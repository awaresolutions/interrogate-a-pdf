import streamlit as st
import pdfplumber
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# --- LLM Integration ---
def run_llama(prompt, model_name="llama3"):
    """
    Sends a prompt to a Llama model hosted by Ollama and returns the response.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except ollama.Ollama.OllamaError as e:
        # This will catch errors if Ollama is not running or the model is not found
        st.error(f"Error: Could not connect to Ollama. Please ensure the server is running and the '{model_name}' model is available. Details: {e}")
        return None

# --- PDF Processing Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    text = ""
    # Use tempfile to handle the uploaded file properly on all OS
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def embed_text_chunks(text, chunk_size=500):
    """Chunks text, creates embeddings, and builds a FAISS index."""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    
    # Use numpy array for FAISS index
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return model, chunks, index

def retrieve_context(query, model, chunks, index, k=3):
    """Retrieves relevant text chunks based on a query."""
    query_embedding = model.encode([query]).astype('float32')
    D, I = index.search(query_embedding, k)
    context = "\n---\n".join([chunks[i] for i in I[0]])
    return context

# --- Streamlit App ---
st.set_page_config(page_title="📄 PDF Chat Agent with Llama", layout="wide")
st.title("PDF Chat Agent with Llama")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting and embedding..."):
        text = extract_text_from_pdf(uploaded_file)
        model, chunks, index = embed_text_chunks(text)
    st.success("PDF processed successfully.")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        context = retrieve_context(query, model, chunks, index)
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, just say "I can't find the answer in the provided document." Do not try to make up an answer.

Context:
{context}

Question: {query}
Answer:"""

        with st.spinner("Generating answer from Llama..."):
            response = run_llama(prompt)

        if response:
            st.subheader("📌 Answer")
            st.markdown(response)

            st.expander("📚 Context Used").write(context)