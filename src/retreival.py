from os import environ
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any
import re

model = SentenceTransformer(model_name_or_path=environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"))

def embed_query(query: str) -> List[float]:

    query_text = f"query: {query}"

    embedding = model.encode(
        query_text,
        normalize_embeddings=True
    )
    
    # Convert to list and return
    return embedding.tolist()

def find_similar_chunks(query_embedding: List[float], collection_name: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Step 2: Find similar chunks using vector search in ChromaDB
    
    Args:
        query_embedding: The embedded query vector
        collection_name: Name of the ChromaDB collection
        n_results: Number of similar chunks to retrieve
        
    Returns:
        List of dictionaries containing similar chunks and their metadata
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
    
    # Get the collection
    collection = client.get_collection(collection_name)
    
    # Convert query embedding to numpy array and ensure it's normalized
    # query_embedding_np = np.array(query_embedding)
    # norm = np.linalg.norm(query_embedding_np, axis=1, keepdims=True)
    # normalized_embedding = query_embedding_np / norm
    
    # # Convert to the format ChromaDB expects
    # query_embedding = normalized_embedding.reshape(1, -1).tolist()
    
    # # Print debug information
    # print(f"\nQuery embedding shape: {np.array(query_embedding).shape}")
    # print(f"Query embedding norm: {np.linalg.norm(np.array(query_embedding))}")
    
    # Query the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    # Format results
    similar_chunks = []
    for i in range(len(results['documents'][0])):
        chunk = {
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity score
        }
        similar_chunks.append(chunk)
        # Print debug information for each result
        print(f"\nResult {i+1}:")
        print(f"Distance: {results['distances'][0][i]}")
        print(f"Similarity Score: {chunk['similarity_score']:.4f}")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content preview: {chunk['content']}")
    
    return similar_chunks

def retrieve_similar_chunks(collection_name:str, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Main function to retrieve similar chunks for a query
    
    Args:
        query: The user's search query
        n_results: Number of similar chunks to retrieve
        
    Returns:
        List of dictionaries containing similar chunks, their metadata, and similarity scores
    """
    # Step 1: Embed the query
    print("Step 1: Embedding query...")
    print(f"Query text: {query}")
    query_embedding = embed_query(query)
    
    # Step 2: Find similar chunks
    print("Step 2: Finding similar chunks...")
    similar_chunks = find_similar_chunks(query_embedding, collection_name, n_results=n_results)
    
    return similar_chunks

def print_results(results: List[Dict[str, Any]]) -> None:
    """Print the retrieval results in a formatted way"""
    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(results, 1):
        if chunk.get("similarity_score") is not None:
            print(f"\n{i}. Similarity Score: {chunk['similarity_score']:.4f}")
        print(f"Content: {chunk['content']}")
        print(f"Metadata: {chunk['metadata']}")

def print_hierarchical_results(results: Dict[str, Dict[str, Any]]) -> None:
    print("\nRetrieved Legal Hierarchy:\n")

    for item in results.values():
        print("=" * 60)
        print(item["doc"])
        print(item["pasal"])
        print("-" * 60)
        print(item["text"])

def hierarchy_key(metadata: Dict[str, Any]) -> str:
    return "|".join([
        metadata.get("doc", ""),
        metadata.get("pasal", ""),
    ])

def pasal_sort_key(pasal: str) -> int:
    match = re.search(r"\d+", pasal)
    return int(match.group()) if match else float("inf")

def get_merged_hierarchical_chunks(
    similar_chunks: List[Dict[str, Any]],
    collection_name: str
) -> Dict[str, Dict[str, Any]]:

    if not similar_chunks:
        return {}

    # Collect unique hierarchy groups
    groups = {}
    for c in similar_chunks:
        meta = c.get("metadata", {})
        key = hierarchy_key(meta)
        groups[key] = meta

    client = chromadb.HttpClient(
        host="localhost",
        port=8001,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    collection = client.get_collection(collection_name)

    merged_results = {}

    # print("\nGroups:")
    # print(groups)
    # print("end groups\n")
    
    for key, meta in groups.items():
        where_clause = {
            "$and": [
                {"doc": meta.get("doc")},
                {"pasal": meta.get("pasal")}
            ]
        }

        results = collection.get(
            where=where_clause,
            include=["documents", "metadatas"]
        )

        def sort_byorderno(m):
            return meta.get("orderno", float("inf"))

        chunks = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: sort_byorderno(x[1])
        )

        merged_text = "\n".join(doc.strip() for doc, _ in chunks)

        merged_results[key] = {
            "doc": meta["doc"],
            "bab": meta["bab"],
            "pasal": meta["pasal"],
            "text": merged_text
        }

    return merged_results

def main():
    # Example usage
    # query = "Apa saja sanksi yang mungkin dikenakan kepada bank?"
    query = "Apa saja tugas dan tanggung jawab OJK?"
    
    print(f"Query: {query}")
    
    results = retrieve_similar_chunks(collection_name="policy_documents", query=query, n_results=5)
    
    print("\nChunks with the same 'pasal':")
    same_pasal_chunks = get_merged_hierarchical_chunks(results, collection_name="policy_documents_nooverlap")
    print_hierarchical_results(same_pasal_chunks)

if __name__ == "__main__":
    main()
