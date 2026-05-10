import os
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class AnswerGenerator:
    # Use gpt-4o-mini — much cheaper per token than gpt-4o.
    # Raise max_tokens only after upgrading OpenRouter credits.
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=400,   # Keep low for free-tier credits; increase when on paid plan
            temperature=0.0
        )
        
        self.system_prompt = """You are a precise question-answering assistant. Answer the user's question 
using ONLY the provided context. Cite your sources using [Source: filename, 
page X] after each claim. If the context is insufficient, say "I cannot 
answer this from the available documents." Do not use outside knowledge."""

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Generates an answer based on the context and returns the answer along with the sources.
        """
        if not context_chunks:
            return "I cannot answer this from the available documents.", []

        # Format context
        context_str = ""
        sources = []
        for chunk in context_chunks:
            metadata = chunk["metadata"]
            source_file = metadata.get("source_file", "Unknown")
            page = metadata.get("page_number", "Unknown")
            source_label = f"[Source: {source_file}, page {page}]"
            context_str += f"{source_label}\n{chunk['text']}\n\n"
            sources.append(source_label)

        # Remove duplicate sources
        sources = list(dict.fromkeys(sources))

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content, sources
        except Exception as e:
            return f"Error generating answer: {e}", sources
