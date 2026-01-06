import streamlit as st
import json
import os
import sys
import chromadb
from chromadb.config import Settings
from os import environ
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path setup
from retreival import retrieve_similar_chunks, get_merged_hierarchical_chunks
from google_llm_response import get_llm_response

# Configuration constants
MODEL_NAME = environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = "policy_documents"

def get_chunks_by_pasal(pasal_numbers):
    """Retrieve and concatenate chunks that have the specified Pasal numbers"""
    try:
        # Connect to ChromaDB
        client = chromadb.HttpClient(
            host="localhost",
            port=8001,
            settings=Settings(allow_reset=True)
        )
        collection = client.get_collection("policy_documents")
        
        # Get all chunks
        results = collection.get()
        
        # Group chunks by Pasal
        pasal_chunks = {}
        for i, metadata in enumerate(results['metadatas']):
            pasal = metadata.get('pasal')
            if pasal in pasal_numbers:
                if pasal not in pasal_chunks:
                    pasal_chunks[pasal] = {
                        'content': [],
                        'metadata': metadata,
                        'pages': set()
                    }
                pasal_chunks[pasal]['content'].append(results['documents'][i])
                pasal_chunks[pasal]['pages'].add(metadata['halaman'])
        
        # Convert grouped chunks to list format
        filtered_chunks = []
        for pasal, data in pasal_chunks.items():
            chunk = {
                'content': ' '.join(data['content']),
                'metadata': data['metadata'],
                'similarity_score': 1.0,  # Since these are exact matches
                'pages': sorted(list(data['pages']))
            }
            filtered_chunks.append(chunk)
            print("PASS===",pasal)
        
        return filtered_chunks
    except Exception as e:
        st.error(f"Error retrieving chunks by Pasal: {str(e)}")
        return []

def main():
    try:
        st.set_page_config(
            page_title="Document Q&A",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("Q&A System")
        st.markdown("""
        Ask questions about the policy documents and get relevant information from the documents.
        The system will retrieve the most relevant chunks of text that answer your question.
        """)
        
        # Sidebar
        with st.sidebar:
            st.header("About")
            st.markdown("""
            This system uses:
            - BGE-M3 for embeddings
            - ChromaDB for vector storage
            - Google API for text generation
            - Streamlit for the UI
            """)
            
            st.header("Settings")
            n_results = st.slider("Number of results to show", 1, 10, 5)
            
            # Add LLM settings
            st.subheader("LLM Settings")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
            top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.1)
        
        # Main content
        query = st.text_input("Enter your question:", placeholder="e.g., Apa saja fungsi OJK?")
        
        if query:
            with st.spinner("Searching for relevant information..."):
                try:
                    # Get initial results
                    results = retrieve_similar_chunks(
                        collection_name=COLLECTION_NAME,
                        query=query,
                        n_results=n_results
                    )
                    
                    if results:
                        st.subheader("Retrieved Information")
                        
                        same_pasal_chunks = get_merged_hierarchical_chunks(results, collection_name="policy_documents_nooverlap")

                        if same_pasal_chunks:
                            st.markdown("### 📑 Relevant Documents")
                            pasal_text = "<br>".join(same_pasal_chunks.keys())
                            st.markdown(f"**Found in:** {pasal_text}", unsafe_allow_html=True)
                            st.markdown("---")
                        
                        for item in same_pasal_chunks.values():
                            st.markdown(f"**Dokumen:** {item["doc"]}**")
                            st.markdown(f"**Pasal:** {item["pasal"]}**")
                            st.markdown(item["text"])
                            st.markdown("-" * 60)
                        
                        # Generate LLM response
                        with st.spinner("Generating response..."):
                            try:
                                response = get_llm_response(
                                    query=query,
                                    chunks=same_pasal_chunks,
                                    temperature=temperature,
                                    top_p=top_p
                                )
                                
                                st.markdown("---")
                                st.markdown("### Generated Response")
                                st.markdown(response)
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")
                    else:
                        st.warning("No relevant information found. Try rephrasing your question.")
                except Exception as e:
                    st.error(f"Error retrieving results: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 