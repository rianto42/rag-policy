from typing import List, Dict, Any
import os
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm_response(
    query: str,
    chunks: Dict[str, Dict[str, Any]],
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Generate an LLM response based on provided context chunks.
    """

    try:
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_AI_API_KEY tidak ditemukan. "
                "Tambahkan ke file .env: GOOGLE_AI_API_KEY=your-api-key"
            )

        # Build context safely
        context_parts: List[str] = []
        # for idx, chunk in enumerate(chunks, start=1):
        #     content = chunk.get("content")
        #     if isinstance(content, str) and content.strip():
        #         context_parts.append(f"{idx}. {content.strip()}")
        
        for item in chunks.values():
            context_parts.append(f"**Dokumen:** {item["doc"]}**")
            context_parts.append(f"**Pasal:** {item["pasal"]}**")
            context_parts.append(item["text"])
            context_parts.append("-" * 60)

        if not context_parts:
            logger.warning("Context kosong")
            return "Maaf, tidak ada informasi yang tersedia untuk menjawab pertanyaan Anda."

        context = "\n".join(context_parts)

        model_name = os.getenv("GOOGLE_AI_MODEL", "gemini-pro")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            timeout=None,
            max_tokens=None
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "Anda adalah asisten AI yang menjawab pertanyaan "
                    "- HANYA berdasarkan konteks yang diberikan, tidak perlu menggunakan kalimat seperti 'berdasarkan konteks yang diberikan'.\n"
                    "- Gunakan Bahasa Indonesia yang baik dan benar.\n"
                    "- Berikan jawaban dengan referensi dokumen dan pasal yang relevan."
                    "- Jika jawaban tidak ada di konteks, katakan 'Informasi yang dibutuhkan untuk menjawab pertanyaan Anda tidak tersedia.'."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Konteks:\n{context}\n\nPertanyaan:\n{query}"
                ),
            ]
        )

        messages = prompt.format_messages(
            context=context,
            query=query,
        )

        response = llm.invoke(messages)

        if not response or not getattr(response, "content", None):
            logger.warning("Response kosong dari LLM")
            return "Maaf, saya tidak dapat menghasilkan jawaban."

        return response.content.strip()

    except Exception as e:
        logger.exception("Gagal menghasilkan response")
        return f"Terjadi kesalahan: {str(e)}"


if __name__ == "__main__":
    test_chunks = [
        {
            "content": "OJK adalah lembaga yang mengatur dan mengawasi sektor jasa keuangan.",
            "metadata": {"pasal": "Pasal 1", "halaman": 1},
        },
        {
            "content": "Fungsi OJK meliputi pengaturan, pengawasan, dan perlindungan konsumen.",
            "metadata": {"pasal": "Pasal 2", "halaman": 2},
        },
    ]

    query = "Apa saja fungsi OJK?"

    result = get_llm_response(
        query=query,
        chunks=test_chunks,
    )

    print("\nPertanyaan:", query)
    print("\nJawaban:", result)
