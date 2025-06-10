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
        
    for level, pattern in patterns.items():
        match = re.match(pattern, line)
        if match:
            # Return only the matched part for hierarchy
            return level, match.group(0).strip(), line
    return None, None, line

def extract_hierarchy(text, current_hierarchy):
    patterns = {
        'bab': r'BAB\s+[IVXLCDM]+(?:\s|$)',
        'bagian': r'Bagian\sKe[a-z]+(?:\s|$)',
        'pasal': r'Pasal\s+\d+(?:\s|$)',
        'ayat': r'\(\d+\)(?:\s|$)',
        'huruf': r'[a-z]\.(?:\s|$)'
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
                        current_hierarchy_value = current_hierarchy[level]
                        current_content = content
                        break
    
    if current_level:
        hierarchy.append((current_level, current_hierarchy_value, current_content.strip()))
    
    return hierarchy

def chunking(pages):
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
                    current_hierarchy['ayat'] = hierarchy_value
                    current_hierarchy['huruf'] = None
                elif level == 'huruf':
                    current_hierarchy['huruf'] = hierarchy_value
                
                chunk = {
                    "content": content,
                    "metadata": {
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
                if(chunk['metadata']['hierarchy']['bab'] != None):
                    chunks.append(chunk)
            break
        else:
            hierarchy = extract_hierarchy(text, current_hierarchy)
            
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
                    current_hierarchy['ayat'] = hierarchy_value
                    current_hierarchy['huruf'] = None
                elif level == 'huruf':
                    current_hierarchy['huruf'] = hierarchy_value
                
                chunk = {
                    "content": content,
                    "metadata": {
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
                if(chunk['metadata']['hierarchy']['bab'] != None):
                    chunks.append(chunk)
    
    return chunks

def save_chunks_to_file(chunks, output_file):
    """Save chunks to a JSON file"""
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert chunks to JSON-serializable format
    serializable_chunks = []
    for chunk in chunks:
        serializable_chunks.append({
            "content": chunk["content"],
            "metadata": {
                "halaman": chunk["metadata"]["halaman"],
                "hierarchy": chunk["metadata"]["hierarchy"]
            }
        })
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(chunks)} chunks to {output_file}")

# Test the code
# pages = load_document("UU21.pdf")
# chunks = chunking(pages)
# print(f"\nFinal number of chunks: {len(chunks)}")
# if chunks:
#     print("\nChunks:")
#     for i, chunk in enumerate(chunks):
#         print(f"\nChunk {i}:")
#         print(f"Metadata: {chunk['metadata']}")
#         print(f"Content: {chunk['content']}...")
    
#     # Save chunks to file
#     save_chunks_to_file(chunks, "output/chunks.json")

