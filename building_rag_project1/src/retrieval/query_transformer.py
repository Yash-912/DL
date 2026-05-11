from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class QueryVariant:
    text: str
    kind: str
    vector: Optional[List[float]] = None


class QueryTransformer:
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=self._get_openrouter_key(),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=256,
        )

    def rewrite(self, query: str) -> str:
        system = "Rewrite the query for retrieval. Preserve intent, remove fluff, expand abbreviations."
        return self._single_output(system, query)

    def step_back(self, query: str) -> str:
        system = "Rewrite the query into a broader, more general version for retrieval."
        return self._single_output(system, query)

    def multi_query(self, query: str, count: int = 3) -> List[str]:
        system = (
            "Generate {count} alternative search queries for the user question. "
            "Return each on its own line, no numbering or quotes."
        ).format(count=count)
        output = self._single_output(system, query)
        variants = [line.strip() for line in output.splitlines() if line.strip()]
        return variants[:count]

    def hyde(self, query: str) -> str:
        system = (
            "Write a short hypothetical passage that would directly answer the question. "
            "Keep it factual and concise."
        )
        return self._single_output(system, query)

    def build_variants(
        self,
        query: str,
        mode: str,
        multi_count: int = 3,
    ) -> List[QueryVariant]:
        variants: List[QueryVariant] = [QueryVariant(text=query, kind="original")]

        if mode in {"rewrite", "all"}:
            variants.append(QueryVariant(text=self.rewrite(query), kind="rewrite"))
        if mode in {"step_back", "all"}:
            variants.append(QueryVariant(text=self.step_back(query), kind="step_back"))
        if mode in {"multi", "all"}:
            for text in self.multi_query(query, count=multi_count):
                variants.append(QueryVariant(text=text, kind="multi"))
        if mode in {"hyde", "all"}:
            variants.append(QueryVariant(text=self.hyde(query), kind="hyde"))

        return variants

    def _single_output(self, system_prompt: str, query: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _get_openrouter_key(self) -> str:
        from os import getenv

        api_key = getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return api_key
