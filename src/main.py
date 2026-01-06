import ingestion
import embedding
import storing
# import retreival

model_name = "BAAI/bge-m3"
chunks_file = "output/chunks.json"
chunks_file_no_overlap = "output/chunks_no_overlap.json"
output_file = "output/chunks_with_embeddings.json"
output_file_no_overlap = "output/chunks_with_embeddings_no_overlap.json"
reprocess = True

if reprocess:
    filename = "POJK11.pdf"
    title = "Penyelenggaran TI oleh Bank Umum"

    # Ingestion
    pages = ingestion.load_document(filename)
    chunks = ingestion.chunking(pages, title)
    print(f"\nFinal number of chunks: {len(chunks)}")
    ingestion.save_chunks_to_file(chunks, chunks_file)

    chunk_no_overlap = ingestion.chunking(pages, title, overlap=0)
    print(f"\nFinal number of chunks no overlap: {len(chunk_no_overlap)}")
    ingestion.save_chunks_to_file(chunk_no_overlap, chunks_file_no_overlap)

    # Embedding
    chunks = embedding.load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} embedding chunks")
    chunks_with_embeddings = embedding.create_embeddings_using_transformers(chunks)
    print(f"Created embeddings for {len(chunks_with_embeddings)} chunks")
    embedding.save_embeddings(chunks_with_embeddings, output_file)
    
    chunks_no_overlap = embedding.load_chunks(chunks_file_no_overlap)
    print(f"Loaded {len(chunks_no_overlap)} embedding chunks no overlap")
    chunks_with_embeddings_no_overlap = embedding.create_embeddings_using_transformers(chunks_no_overlap)
    print(f"Created embeddings for {len(chunks_with_embeddings_no_overlap)} chunks no overlap")
    embedding.save_embeddings(chunks_with_embeddings_no_overlap, output_file_no_overlap)

    # Store into DB
    collection = storing.store_in_chroma(chunks_with_embeddings, collection_name="policy_documents", recreate_collection=False)
    collection = storing.store_in_chroma(chunks_with_embeddings_no_overlap,collection_name="policy_documents_nooverlap", recreate_collection=False)

    print("Done processing and storing data.")

# Test Retreival
# query = "Apakah boleh menempatkan pusat data di luar negeri?"
# print(f"Query: {query}")
# results = retreival.retrieve_similar_chunks(collection_name="policy_documents", query=query, n_results=5)
# same_pasal_chunks = retreival.get_merged_hierarchical_chunks(similar_chunks=results, collection_name="policy_documents_nooverlap")
# retreival.print_hierarchical_results(same_pasal_chunks)
# retreival.print_results(results)