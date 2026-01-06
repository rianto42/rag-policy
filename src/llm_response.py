from typing import List, Dict, Any, Optional
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
from retreival import retrieve_similar_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMResponder:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        try:
            # Force CPU usage
            self.device = "cpu"
            logger.info(f"Initializing model {model_name} on {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )
            
            # Load model with CPU optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # Set model to evaluation mode
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
        
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved chunks into a context string
        
        Args:
            chunks: List of chunks with content and metadata
            
        Returns:
            Formatted context string
        """
        if not chunks:
            logger.warning("No chunks provided for context formatting")
            return ""
            
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            try:
                content = chunk.get('content', '').strip()
                if content:
                    context_parts.append(f"{i}.{content}\n")
                
            except Exception as e:
                logger.error(f"Error formatting chunk: {str(e)}")
                continue
        
        return "\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for the model
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            Formatted prompt for the model
        """
        if not context:
            logger.warning("Empty context provided for prompt generation")
            
        return f"""Anda adalah asisten yang menjawab pertanyaan berdasarkan konteks yang diberikan. 
Jawaban harus:
1. Hanya berdasarkan informasi dari konteks
2. Jelas dan ringkas
3. Dalam Bahasa Indonesia
4. Tidak menambahkan informasi di luar konteks

Konteks:
{context}

Pertanyaan: {query}
"""
    
    def generate_response(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        max_length: int = 2000,
        temperature: float = 0.3,  # Lower temperature for more focused responses
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response using the model
        
        Args:
            query: User's question
            chunks: List of retrieved chunks
            max_length: Maximum length of the generated response
            temperature: Controls randomness (0.0 to 1.0)
            top_p: Controls diversity via nucleus sampling
            
        Returns:
            Generated response
        """
        try:
            # Format context from chunks
            context = self.format_context(chunks)
            
            # Generate prompt
            prompt = self.generate_prompt(query, context)
            logger.info(f"Generated prompt: {prompt[:200]}...")  # Log first 200 chars
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # Adjust based on model's context window
                return_attention_mask=True
            )
            
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            logger.info(f"Generating response for query: {query[:50]}...")
            
            # Generate response with CPU-optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,  # max_new_tokens for causal LMs
                    min_new_tokens=20,  # Minimum response length
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the full output
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response (for causal LMs, output includes input)
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                # Fallback: just use the decoded output
                response = full_response.strip()
            
            logger.info(f"Generated response: {response[:200]}...")
            
            # Check if response is empty or too short
            if not response or len(response.strip()) < 10:
                logger.warning("Generated response is too short or empty")
                return "Maaf, saya tidak dapat menghasilkan jawaban yang memadai. Mohon coba pertanyaan lain atau periksa konteks yang diberikan."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

def get_llm_response(
    query: str, 
    chunks: List[Dict[str, Any]], 
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.3,
    top_p: float = 0.9
) -> str:
    """
    Convenience function to get model response
    
    Args:
        query: User's question
        chunks: List of retrieved chunks
        model_name: Name of the HuggingFace model to use
        temperature: Controls randomness (0.0 to 1.0)
        top_p: Controls diversity via nucleus sampling
        
    Returns:
        Generated response
    """
    try:
        responder = LLMResponder(model_name)
        return responder.generate_response(
            query, 
            chunks,
            temperature=temperature,
            top_p=top_p
        )
    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    try:
        query = "Dari mana sumber anggaran OJK?"
        chunk_retreived = retrieve_similar_chunks(
            model="BAAI/bge-m3",
            collection_name="policy_documents",
            query=query,
            n_results=3
        )

        response = get_llm_response(
            query=query,
            chunks=chunk_retreived,
            temperature=0.3,
            top_p=0.9
        )
        print(f"\nQuery: {query}")
        print(f"\nResponse: {response}")

    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise