import fitz
import re
import json
from pathlib import Path

def load_document(docpath):
    doc = fitz.open(docpath)
    pages = []
    for page_num, page in enumerate(doc, 1):
        # Get text with line breaks preserved
        text = page.get_text("text", sort=True)
        # Clean up extra spaces within each line but preserve line breaks
        text = '\n'.join(line.strip() for line in text.split('\n'))
        # Extract page number if it exists in the format "- X -"
        page_num_match = re.search(r'-\s*(\d+)\s*-', text)
        if page_num_match:
            actual_page = int(page_num_match.group(1))
        else:
            actual_page = page_num
        pages.append((text, actual_page))
    return pages

def evaluate_line(line, patterns):
    """Evaluate a single line against all patterns"""
    # Skip page number markers
    if re.match(r'-\s*\d+\s*-$', line.strip()):
        return None, None, None
    
    # Skip lines ending with three consecutive dots
    if line.strip().endswith('...'):
        return None, None, None
        
    for level, pattern in patterns.items():
        match = re.match(pattern, line)
        if match:
            # Return only the matched part for hierarchy
            return level, match.group(0).strip(), line
    return None, None, line

def extract_hierarchy(text, current_hierarchy):
    # patterns = {
    #     'bab': r'BAB\s+[IVXLCDM]+(?:\s|$)',
    #     'bagian': r'Bagian\sKe[a-z]+(?:\s|$)',
    #     'pasal': r'Pasal\s+\d+(?:\s|$)',
    #     'ayat': r'\(\d+\)(?:\s|$)',
    #     'huruf': r'[a-z]\.(?:\s|$)'
    # }
    patterns = {
        'bab': r'^BAB\s+[IVXLCDM]+$',                 # Matches exactly 'BAB II', no trailing text
        'bagian': r'^Bagian\s+Ke[a-z]+$',             # Matches 'Bagian Keempat', nothing after
        'pasal': r'^Pasal\s+\d+$',                    # Matches 'Pasal 20' only if ends right after number
        'ayat': r'\(\d+\)(?:\s|$)',                   # Matches '(1)', '(2)' exactly, nothing else
        'huruf': r'[a-z]\.(?:\s|$)'                   # Matches 'a.', 'b.', etc. with no space or text after
    }
    
    hierarchy = []
    current_level = None
    current_content = ""
    current_hierarchy_value = None
    
    # Split text into lines and process each line
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for "Ditetapkan di" to stop processing
        if "Ditetapkan di" in line or "Disahkan di" in line:
            # Finish the last chunk if there is one
            if current_level:
                hierarchy.append((current_level, current_hierarchy_value, current_content.strip()))
            break
            
        level, hierarchy_value, content = evaluate_line(line, patterns)
        
        # Skip if content is None (page number marker)
        if content is None:
            continue
            
        if level:
            if current_level:
                hierarchy.append((current_level, current_hierarchy_value, current_content.strip()))
            current_level = level
            current_hierarchy_value = hierarchy_value
            current_content = content
        else:
            if current_level:
                current_content += "\n" + content
            # If no level found and we have a current hierarchy, use the last known level
            elif any(current_hierarchy.values()):
                # Find the last non-None level in current_hierarchy
                for level in ['huruf', 'ayat', 'pasal', 'bagian', 'bab']:
                    if current_hierarchy[level] is not None:
                        current_level = level
                        current_hierarchy_value = current_hierarchy[level].replace("Huruf ","").replace("Ayat ","")
                        current_content = content
                        break
    
    if current_level:
        hierarchy.append((current_level, current_hierarchy_value, current_content.strip()))
    
    return hierarchy

def split_content(content, max_length=1024, overlap=100):
    """
    Split content into chunks with overlap

    Args:
        content (str): Text to split
        max_length (int): Max length per chunk
        overlap (int): Number of characters to overlap between chunks

    Returns:
        List[str]: List of overlapping chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', content)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Handle case where a single sentence exceeds max_length
        if len(sentence) > max_length:
            # First, save current_chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split the oversized sentence into smaller chunks
            # Split by words to avoid breaking words
            words = sentence.split()
            temp_chunk = ""
            
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_length:
                    if temp_chunk:
                        temp_chunk += " " + word
                    else:
                        temp_chunk = word
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                        # Start new chunk with overlap
                        if overlap > 0 and len(temp_chunk) >= overlap:
                            temp_chunk = temp_chunk[-overlap:] + " " + word
                        else:
                            temp_chunk = word
                    else:
                        # Even a single word is too long, add it anyway
                        temp_chunk = word
            
            # Set current_chunk to the last temp_chunk
            current_chunk = temp_chunk
            continue
        
        # Normal processing for sentences that fit
        if len(current_chunk) + len(sentence) + 1 > max_length:  # +1 for space
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) >= overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def combine_hierarchy_info_to_content(content, hierarchy, overlap = 100):
    # Remove duplicated Pasal header if present
    if hierarchy.get('pasal') and content.startswith(hierarchy['pasal']):
        cleaned_content = content[len(hierarchy['pasal']):].strip()
    else:
        cleaned_content = content.strip()

    cleaned_content = cleaned_content.replace('\n', ' ')

    # Build hierarchy prefix ONCE
    hierarchy_lines = []
    for key in ['bab', 'pasal']:
        if hierarchy.get(key):
            hierarchy_lines.append(hierarchy[key])

    hierarchy_text = " > ".join(hierarchy_lines)

    # Chunk ONLY the content
    content_chunks = split_content(
        cleaned_content,
        max_length=300,
        overlap=overlap
    )

    # Prepend hierarchy to EVERY chunk
    final_chunks = []
    for chunk in content_chunks:
        # full_chunk = f"{hierarchy_text}\n{chunk}"
        full_chunk = f"{chunk}"
        final_chunks.append(full_chunk)

    return final_chunks

def chunking(pages, document_name, overlap = 100):
    chunks = []
    current_hierarchy = {
        'bab': None,
        'bagian': None,
        'pasal': None,
        'ayat': None,
        'huruf': None
    }
    
    for text, page_num in pages:
        # Check if "Ditetapkan di" is in the text
        if "Ditetapkan di" in text or "Disahkan di" in text:
            # Process only the text before "Ditetapkan di"
            if "Ditetapkan di" in text:
                text = text.split("Ditetapkan di")[0]
            elif "Disahkan di" in text:
                text = text.split("Disahkan di")[0]
            hierarchy = extract_hierarchy(text, current_hierarchy)
            
            # Process the last chunk and break
            for level, hierarchy_value, content in hierarchy:
                if level == 'bab':
                    current_hierarchy = {
                        'bab': hierarchy_value,
                        'bagian': None,
                        'pasal': None,
                        'ayat': None,
                        'huruf': None
                    }
                elif level == 'bagian':
                    current_hierarchy['bagian'] = hierarchy_value
                    current_hierarchy['pasal'] = None
                    current_hierarchy['ayat'] = None
                    current_hierarchy['huruf'] = None
                elif level == 'pasal':
                    current_hierarchy['pasal'] = hierarchy_value
                    current_hierarchy['ayat'] = None
                    current_hierarchy['huruf'] = None
                elif level == 'ayat':
                    current_hierarchy['ayat'] = 'Ayat ' + hierarchy_value.strip('.')
                    current_hierarchy['huruf'] = None
                elif level == 'huruf':
                    current_hierarchy['huruf'] = 'Huruf ' + hierarchy_value.strip('.')
                
                # Get content chunks
                content_chunks = combine_hierarchy_info_to_content(content, current_hierarchy, overlap)
                
                # Create a chunk for each content piece
                for content_chunk in content_chunks:
                    chunk = {
                        "content": content_chunk,
                        "metadata": {
                            "doc": document_name,
                            "halaman": page_num,
                            "hierarchy": {
                                'bab': current_hierarchy['bab'],
                                'bagian': current_hierarchy['bagian'],
                                'pasal': current_hierarchy['pasal'],
                                'ayat': current_hierarchy['ayat'],
                                'huruf': current_hierarchy['huruf']
                            }
                        }
                    }
                    if(chunk['metadata']['hierarchy']['bab'] != None) and chunk["content"] != "":
                        chunks.append(chunk)
            break
        else:
            # Create a copy of current hierarchy for this page
            page_hierarchy = current_hierarchy.copy()
            hierarchy = extract_hierarchy(text, page_hierarchy)
            
            for level, hierarchy_value, content in hierarchy:
                if level == 'bab':
                    page_hierarchy = {
                        'bab': hierarchy_value,
                        'bagian': None,
                        'pasal': None,
                        'ayat': None,
                        'huruf': None
                    }
                elif level == 'bagian':
                    page_hierarchy['bagian'] = hierarchy_value
                    page_hierarchy['pasal'] = None
                    page_hierarchy['ayat'] = None
                    page_hierarchy['huruf'] = None
                elif level == 'pasal':
                    page_hierarchy['pasal'] = hierarchy_value
                    page_hierarchy['ayat'] = None
                    page_hierarchy['huruf'] = None
                elif level == 'ayat':
                    page_hierarchy['ayat'] = 'Ayat ' + hierarchy_value.strip('.')
                    page_hierarchy['huruf'] = None
                elif level == 'huruf':
                    page_hierarchy['huruf'] = 'Huruf ' + hierarchy_value.strip('.')
                
                # Get content chunks
                content_chunks = combine_hierarchy_info_to_content(content, page_hierarchy, overlap)
                
                # Create a chunk for each content piece
                for content_chunk in content_chunks:
                    chunk = {
                        "content": content_chunk,
                        "metadata": {
                            "doc": document_name,
                            "halaman": page_num,
                            "hierarchy": {
                                'bab': page_hierarchy['bab'],
                                'bagian': page_hierarchy['bagian'],
                                'pasal': page_hierarchy['pasal'],
                                'ayat': page_hierarchy['ayat'],
                                'huruf': page_hierarchy['huruf']
                            }
                        }
                    }
                    if(chunk['metadata']['hierarchy']['bab'] != None) and chunk["content"] != "":
                        chunks.append(chunk)
            
            # Update the current hierarchy with the last valid hierarchy from this page
            if page_hierarchy['bab'] is not None:
                current_hierarchy = page_hierarchy
    
    return chunks

def merge_chunks_with_same_metadata(chunks):
    """
    Merge chunks that have identical metadata by combining their content.
    Ignores the 'halaman' field when comparing metadata.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of merged chunks
    """
    # Create a dictionary to store chunks by their metadata
    merged_chunks = {}
    
    for chunk in chunks:
        # Convert metadata to a tuple of tuples for hashing, excluding halaman
        metadata_key = (
            ('doc', chunk['metadata']['doc']),
            ('hierarchy', (
                ('bab', chunk['metadata']['hierarchy']['bab']),
                ('bagian', chunk['metadata']['hierarchy']['bagian']),
                ('pasal', chunk['metadata']['hierarchy']['pasal']),
                ('ayat', chunk['metadata']['hierarchy']['ayat']),
                ('huruf', chunk['metadata']['hierarchy']['huruf'])
            ))
        )
        
        # Convert to string for dictionary key
        metadata_key = str(metadata_key)
        
        if metadata_key in merged_chunks:
            # If chunk with same metadata exists, append content
            merged_chunks[metadata_key]['content'] += ' ' + chunk['content']
            # Keep the lowest page number
            merged_chunks[metadata_key]['metadata']['halaman'] = min(
                merged_chunks[metadata_key]['metadata']['halaman'],
                chunk['metadata']['halaman']
            )
        else:
            # If it's a new metadata, add the chunk
            merged_chunks[metadata_key] = chunk.copy()
    
    # add chunk length info for merged_chunks
    for key in merged_chunks:
        merged_chunks[key]['metadata']['chunk_length'] = len(merged_chunks[key]['content'])
    # Convert dictionary back to list
    return list(merged_chunks.values())

def save_chunks_to_file(chunks, output_file):
    """Save chunks to a JSON file"""
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # print(f"Saved {len(merged_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    pages = load_document("UU21.pdf")
    chunks = chunking(pages, "Undang-Undang Nomor 21 Tahun 2011 Tentang OJK", overlap=100)
    print(f"\nFinal number of chunks: {len(chunks)}")
    if chunks:
        # print("\nChunks:")
        # for i, chunk in enumerate(chunks):
        #     print(f"\nChunk {i}:")
        #     print(f"Metadata: {chunk['metadata']}")
        #     print(f"Content: {chunk['content']}...")

        # Merge chunks with identical metadata
        #chunks = merge_chunks_with_same_metadata(chunks)
        # print(f"\nMerged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        
        # Save chunks to file
        save_chunks_to_file(chunks, "output/chunks.json")

    # get maximum chunks info from metadata
    # max_chunk = max(chunks, key=lambda x: x['metadata']['chunk_length'])
    # print(f"\nMaximum chunk length: {max_chunk['metadata']['chunk_length']}")

    chunk_no_overlap = chunking(pages, "Undang-Undang Nomor 21 Tahun 2011 Tentang OJK", overlap=0)
    save_chunks_to_file(chunk_no_overlap, "output/chunks_no_overlap.json")