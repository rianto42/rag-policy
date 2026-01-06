import chromadb
from chromadb.config import Settings

def verify_collection():
    # Initialize ChromaDB client
    client = chromadb.HttpClient(
        host="localhost",
        port=8001,
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    
    # List all collections
    collections = client.list_collections()
    print("\nAvailable collections:")
    for collection in collections:
        print(f"\nCollection: {collection.name}")
        print(f"Number of documents: {collection.count()}")
        
        # Get a sample of documents
        results = collection.get(
            limit=1,
            include=['documents', 'metadatas']
        )
        if results['documents']:
            print("\nSample document:")
            print(f"Content: {results['documents'][0][:200]}...")
            print(f"Metadata: {results['metadatas'][0]}")

if __name__ == "__main__":
    verify_collection() 