import chromadb
import shutil
import os
from chromadb.config import Settings

CHROMA_DIR = "/chroma_policyrag"

# Close all collections
client = chromadb.HttpClient(
        host="localhost",
        port=8001,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
for c in client.list_collections():
    client.delete_collection(c.name)

print("ChromaDB fully reset.")

# Check collection
collection = client.get_or_create_collection(
    name="policy_documents",
    # get_or_create=True,
    metadata={"description": "Policy documents with embeddings"}
)
print(f"Connected to collection: {"policy_documents"} with size {collection.count()}")

collection_no_overlap = client.get_or_create_collection(
    name="policy_documents_nooverlap",
    # get_or_create=True,
    metadata={"description": "Policy documents with embeddings no overlap"}
)
print(f"Connected to collection: policy_documents_nooverlap with size {collection_no_overlap.count()}")