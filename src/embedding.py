from os import environ
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import gc

# Initialize the model
# model = SentenceTransformer(environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"))

def load_chunks(chunks_file):
    """Load chunks from a JSON file"""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit length (L2 normalization)
    
    Args:
        embeddings: Tensor of shape (batch_size, hidden_size) containing embeddings
        
    Returns:
        Normalized embeddings of the same shape
    """
    # Calculate L2 norm for each embedding
    norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    # Normalize by dividing by the norm
    normalized_embeddings = embeddings / norm
    return normalized_embeddings

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling - take average of all token embeddings for each text in the batch separately
    and normalize the resulting embeddings
    
    Args:
        model_output: Output from BERT model containing token embeddings
        attention_mask: Binary mask indicating which tokens are actual words (1) and which are padding (0)
    
    Returns:
        Tensor of shape (batch_size, hidden_size) containing normalized mean-pooled embeddings for each text
    """
    # Get token embeddings from the last hidden state
    token_embeddings = model_output[0]  # Shape: (batch_size, sequence_length, hidden_size)
    
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum the embeddings for each text in the batch
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)  # Shape: (batch_size, hidden_size)
    
    # Count the number of tokens for each text (excluding padding)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9).unsqueeze(-1)  # Shape: (batch_size, 1)
    
    # Calculate mean embeddings for each text
    mean_embeddings = sum_embeddings / sum_mask  # Shape: (batch_size, hidden_size)
    
    # Normalize the embeddings
    normalized_embeddings = normalize_embeddings(mean_embeddings)
    
    return normalized_embeddings

def create_embeddings(chunks, model_name):
    """
    Create embeddings for each chunk using IndoBERT model
    
    Args:
        chunks: List of dictionaries containing content and metadata
        model_name: Name of the IndoBERT model to use
    
    Returns:
        List of dictionaries containing content, metadata, and embeddings
    """
    # Initialize the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode
    
    # Prepare texts for embedding
    texts = [f"passage: {chunk['content']}" for chunk in chunks]
    
    # Generate embeddings
    chunks_with_embeddings = []
    
    # Process in batches to avoid memory issues
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = chunks[i:i + batch_size]  # Get corresponding chunks for this batch
        
        # Tokenize and prepare input
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        
        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform mean pooling and normalization
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Convert embeddings to numpy arrays for storage
        embeddings_np = embeddings.cpu().numpy()
        
        # Ensure embeddings are normalized
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        
        # Print debug information for first batch
        if i == 0:
            print("\nDebug information for first batch:")
            print(f"Embedding shape: {normalized_embeddings.shape}")
            print(f"Embedding norms: {norms.flatten()}")
        
        # Combine chunks with their embeddings
        for chunk, embedding in zip(batch_chunks, normalized_embeddings):
            chunk_with_embedding = {
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
            }
            chunks_with_embeddings.append(chunk_with_embedding)
    
    return chunks_with_embeddings

def create_embeddings_using_transformers(chunks, batch_size = 8):
    model = SentenceTransformer(environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"))
    # Prepare texts for embedding
    texts = [f"passage: {chunk['content']}" for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    # Debug info
    print("\nDebug information:")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Combine chunks with their embeddings
    chunks_with_embeddings = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_with_embedding = {
            'content': chunk['content'],
            'metadata': chunk['metadata'],
            'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
        }
        chunks_with_embeddings.append(chunk_with_embedding)
    
    del model
    gc.collect()

    return chunks_with_embeddings

def save_embeddings(chunks_with_embeddings, output_file):
    """Save chunks with embeddings to a JSON file"""
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(chunks_with_embeddings)} chunks with embeddings to {output_file}")

if __name__ == "__main__":
    # Example usage
    chunks_file = "output/chunks.json"
    output_file = "output/chunks_with_embeddings.json"
    
    # Load chunks
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create embeddings
    chunks_with_embeddings = create_embeddings_using_transformers(chunks)
    print(f"Created embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Save results
    save_embeddings(chunks_with_embeddings, output_file)

    #### No Overlap file
    chunks_no_overlap_file = "output/chunks_no_overlap.json"
    output_no_overlap_file = "output/chunks_with_embeddings_no_overlap.json"
    
    # Load chunks
    chunks_no_overlap = load_chunks(chunks_no_overlap_file)
    print(f"Loaded {len(chunks_no_overlap)} chunks")
    
    # Create embeddings
    chunks_no_overlap_with_embeddings = create_embeddings_using_transformers(chunks_no_overlap)
    print(f"Created embeddings for {len(chunks_no_overlap_with_embeddings)} chunks")
    
    # Save results
    save_embeddings(chunks_no_overlap_with_embeddings, output_no_overlap_file)