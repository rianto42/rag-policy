import json
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Union

def load_chunks(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

def generate_doc_id(chunk: Dict[str, Any], index: int) -> str:
    """Generate a unique ID based on document hierarchy"""
    doc_name = chunk['metadata']['doc']
    bab = chunk['metadata']['hierarchy']['bab']
    pasal = chunk['metadata']['hierarchy']['pasal']
    ayat = chunk['metadata']['hierarchy']['ayat']
    huruf = chunk['metadata']['hierarchy']['huruf']
    
    # Create a hierarchical ID
    id_parts = [doc_name]
    if bab: id_parts.append(bab)
    if pasal: id_parts.append(pasal)
    if ayat: id_parts.append(ayat)
    if huruf: id_parts.append(huruf)
    
    return "_".join(id_parts) + f"_{index}"

def clean_metadata_value(value: Any) -> Union[str, int, float, bool]:
    """Convert metadata value to a type that ChromaDB can handle"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def prepare_metadata(chunk_order_no: int, chunk: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """Prepare metadata with values that ChromaDB can handle"""
    hierarchy = chunk['metadata']['hierarchy']
    return {
        "doc": clean_metadata_value(chunk['metadata']['doc']),
        "halaman": clean_metadata_value(chunk['metadata']['halaman']),
        "bab": clean_metadata_value(hierarchy['bab']),
        "bagian": clean_metadata_value(hierarchy['bagian']),
        "pasal": clean_metadata_value(hierarchy['pasal']),
        "ayat": clean_metadata_value(hierarchy['ayat']),
        "huruf": clean_metadata_value(hierarchy['huruf']),
        "orderno": chunk_order_no
    }

def store_in_chroma(chunks: List[Dict[str, Any]], collection_name: str, batch_size: int = 100, recreate_collection: bool = True) -> chromadb.Collection:
    """
    Store chunks with embeddings in ChromaDB
    
    Args:
        chunks: List of dictionaries containing content, metadata, and embeddings
        collection_name: Name of the ChromaDB collection
        batch_size: Number of chunks to process at once
    
    Returns:
        ChromaDB collection object
    """
    # Initialize ChromaDB client
    client = chromadb.HttpClient(
        host="localhost",
        port=8001,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    if recreate_collection:
        try:
            client.delete_collection(name=collection_name)
            print("Collection Deleted")
        except:
            pass
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        # get_or_create=True,
        metadata={"description": "Policy documents with embeddings"}
    )
    print(f"Connected to collection: {collection_name} with size {collection.count()}")
    
    # Process in batches
    total_chunks = len(chunks)
    chunk_order_no = 0
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = []
        batch_embeddings = []
        batch_documents = []
        batch_metadatas = []
        
        for j, chunk in enumerate(batch_chunks):

            # Convert embedding to numpy array if it's not already
            embedding = np.array(chunk['embedding'])
            batch_embeddings.append(embedding.tolist())
            
            # Prepare metadata with cleaned values
            metadata = prepare_metadata(chunk_order_no, chunk)
            
            # Add to batch
            batch_ids.append(generate_doc_id(chunk, i + j))
            batch_documents.append(chunk['content'])
            batch_metadatas.append(metadata)

            chunk_order_no += 1
        
        # Add batch to collection
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
       
        print(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
    
    print(f"\nSuccessfully stored {total_chunks} documents in ChromaDB")
    print(f"Collection size: {collection.count()} documents")
    
    return collection

def print_results(results: Dict[str, Any]) -> None:
    """Print query results in a formatted way"""
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n{i+1}. Document:")
        print(f"Content: {doc[:200]}...")
        print(f"Metadata: {metadata}")

def test_queries(collection: chromadb.Collection, chunks: List[Dict[str, Any]]) -> None:
    """Test various query scenarios"""
    print("\nRunning query tests:")
    
    # Test 1: Similarity search
    print("\n1. Similarity Search:")
    results = collection.query(
        query_embeddings=[chunks[0]['embedding']],
        n_results=3
    )
    print_results(results)
    
    # Test 2: Filter by metadata
    print("\n2. Filter by Metadata:")
    results = collection.query(
        query_embeddings=[chunks[0]['embedding']],
        n_results=3,
        # where={"bab": {"$eq": clean_metadata_value(chunks[0]['metadata']['hierarchy']['bab'])}}
        where={"bab": {"$eq": "BAB I"}}
    )
    print_results(results)
    
    # Test 3: Combined search
    print("\n3. Combined Search:")
    bab_value = "BAB I"
    pasal_value = "Pasal 1"
    
    where_clause = {"$and": []}
    if bab_value:
        where_clause["$and"].append({"bab": {"$eq": bab_value}})
    if pasal_value:
        where_clause["$and"].append({"pasal": {"$eq": pasal_value}})
    
    results = collection.query(
        query_embeddings=[chunks[0]['embedding']],
        n_results=3,
        where=where_clause
    )
    print_results(results)

def main():
    # Load chunks with embeddings
    chunks_file = "output/chunks_with_embeddings.json"
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks with embeddings")
    collection = store_in_chroma(chunks,collection_name="policy_documents", recreate_collection=True)
    
    chunks_file_no_overlap = "output/chunks_with_embeddings_no_overlap.json"
    chunks_no_overlap = load_chunks(chunks_file_no_overlap)
    print(f"Loaded {len(chunks_no_overlap)} chunks no overlap")
    collection = store_in_chroma(chunks_no_overlap,collection_name="policy_documents_nooverlap", recreate_collection=True)

    # Run tests
    # test_queries(collection, chunks)

if __name__ == "__main__":
    main()
