import streamlit as st
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
import ingestion
import embedding
import storing

# Configuration constants
OUTPUT_DIR = "output"
TEMP_DIR = "temp_uploads"
COLLECTION_NAME = "policy_documents"
COLLECTION_NAME_NO_OVERLAP = "policy_documents_nooverlap"

def ensure_directories():
    """Ensure output and temp directories exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

def process_pdf_file(uploaded_file, document_title, recreate_collection=False):
    """
    Process uploaded PDF file through ingestion, embedding, and storing
    
    Args:
        uploaded_file: Streamlit uploaded file object
        document_title: Title of the document
        recreate_collection: Whether to recreate the collection
    
    Returns:
        dict: Processing results and statistics
    """
    results = {
        "success": False,
        "chunks_count": 0,
        "chunks_no_overlap_count": 0,
        "error": None
    }
    
    try:
        # Save uploaded file temporarily
        ensure_directories()
        temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Step 1: Ingestion
        with st.spinner("📄 Loading and processing PDF document..."):
            pages = ingestion.load_document(temp_file_path)
            st.success(f"✓ Loaded {len(pages)} pages from document")
        
        with st.spinner("✂️ Chunking document with overlap..."):
            chunks = ingestion.chunking(pages, document_title, overlap=100)
            results["chunks_count"] = len(chunks)
            st.success(f"✓ Created {len(chunks)} chunks with overlap")
        
        with st.spinner("✂️ Chunking document without overlap..."):
            chunks_no_overlap = ingestion.chunking(pages, document_title, overlap=0)
            results["chunks_no_overlap_count"] = len(chunks_no_overlap)
            st.success(f"✓ Created {len(chunks_no_overlap)} chunks without overlap")
        
        # Generate unique filenames based on document title
        safe_title = "".join(c for c in document_title if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        chunks_file = os.path.join(OUTPUT_DIR, f"{safe_title}_chunks.json")
        chunks_file_no_overlap = os.path.join(OUTPUT_DIR, f"{safe_title}_chunks_no_overlap.json")
        output_file = os.path.join(OUTPUT_DIR, f"{safe_title}_chunks_with_embeddings.json")
        output_file_no_overlap = os.path.join(OUTPUT_DIR, f"{safe_title}_chunks_with_embeddings_no_overlap.json")
        
        # Save chunks
        ingestion.save_chunks_to_file(chunks, chunks_file)
        ingestion.save_chunks_to_file(chunks_no_overlap, chunks_file_no_overlap)
        
        # Step 2: Embedding
        with st.spinner("🔢 Creating embeddings for chunks with overlap..."):
            chunks_with_embeddings = embedding.create_embeddings_using_transformers(chunks)
            embedding.save_embeddings(chunks_with_embeddings, output_file)
            st.success(f"✓ Created embeddings for {len(chunks_with_embeddings)} chunks")
        
        with st.spinner("🔢 Creating embeddings for chunks without overlap..."):
            chunks_with_embeddings_no_overlap = embedding.create_embeddings_using_transformers(chunks_no_overlap)
            embedding.save_embeddings(chunks_with_embeddings_no_overlap, output_file_no_overlap)
            st.success(f"✓ Created embeddings for {len(chunks_with_embeddings_no_overlap)} chunks without overlap")
        
        # Step 3: Store in ChromaDB
        with st.spinner("💾 Storing chunks with overlap in vector database..."):
            collection = storing.store_in_chroma(
                chunks_with_embeddings,
                collection_name=COLLECTION_NAME,
                recreate_collection=recreate_collection
            )
            st.success(f"✓ Stored {len(chunks_with_embeddings)} chunks in collection '{COLLECTION_NAME}'")
        
        with st.spinner("💾 Storing chunks without overlap in vector database..."):
            collection_no_overlap = storing.store_in_chroma(
                chunks_with_embeddings_no_overlap,
                collection_name=COLLECTION_NAME_NO_OVERLAP,
                recreate_collection=recreate_collection
            )
            st.success(f"✓ Stored {len(chunks_with_embeddings_no_overlap)} chunks in collection '{COLLECTION_NAME_NO_OVERLAP}'")
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        results["success"] = True
        results["collection_name"] = COLLECTION_NAME
        results["collection_name_no_overlap"] = COLLECTION_NAME_NO_OVERLAP
        results["files"] = {
            "chunks_file": chunks_file,
            "chunks_file_no_overlap": chunks_file_no_overlap,
            "output_file": output_file,
            "output_file_no_overlap": output_file_no_overlap
        }
        
    except Exception as e:
        results["error"] = str(e)
        st.error(f"❌ Error processing file: {str(e)}")
    
    return results

def main():
    st.set_page_config(
        page_title="Upload Document",
        page_icon="📤",
        layout="wide"
    )
    
    st.title("📤 Upload Document")
    st.markdown("""
    Upload a PDF document to process and add it to the vector database.
    The document will be:
    1. **Ingested** - Extracted and chunked based on document structure
    2. **Embedded** - Converted to vector embeddings
    3. **Stored** - Added to ChromaDB for retrieval
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.info(f"**Collection Names:**\n- {COLLECTION_NAME}\n- {COLLECTION_NAME_NO_OVERLAP}")
        
        recreate_collection = st.checkbox(
            "Recreate Collection",
            value=False,
            help="If checked, will delete and recreate the collection. Use with caution!"
        )
        
        st.markdown("---")
        st.header("ℹ️ Information")
        st.markdown("""
        **Processing Steps:**
        1. Document is loaded and pages are extracted
        2. Text is chunked based on document hierarchy (BAB, Pasal, Ayat, etc.)
        3. Chunks are converted to embeddings using BGE-M3 model
        4. Embeddings are stored in ChromaDB
        
        **Note:** Processing may take several minutes depending on document size.
        """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to process"
    )
    
    document_title = st.text_input(
        "Document Title",
        placeholder="e.g., Peraturan OJK Nomor 11 Tahun 2021",
        help="Enter a descriptive title for this document"
    )
    
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("❌ Please upload a PDF file first")
            return
        
        if not document_title or not document_title.strip():
            st.error("❌ Please enter a document title")
            return
        
        # Process the file
        with st.container():
            st.markdown("### Processing Status")
            
            results = process_pdf_file(
                uploaded_file,
                document_title.strip(),
                recreate_collection
            )
            
            if results["success"]:
                st.markdown("---")
                st.markdown("### ✅ Processing Complete!")
                
                # Display statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chunks (with overlap)", results["chunks_count"])
                    st.metric("Collection Name", results["collection_name"])
                
                with col2:
                    st.metric("Chunks (no overlap)", results["chunks_no_overlap_count"])
                    st.metric("No Overlap Collection", results["collection_name_no_overlap"])
                
                # Display saved files
                st.markdown("### 📁 Saved Files")
                st.json(results["files"])
                
                st.success("🎉 Document successfully processed and added to the vector database!")
                st.info("💡 You can now use the chat interface to query this document.")
            else:
                st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()

