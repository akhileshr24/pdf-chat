
import json
import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class LLMGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.answer_llm = ChatGroq(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0.2,
            max_tokens=700,
        )

        self.verification_llm = ChatGroq(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=0,
            max_tokens=900,
        )

        # Answer generation chain
        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Answer using only the provided PDF context. "
                "Do not use outside knowledge or invent facts. "
                "If the answer is unavailable, say you don't know "
                "based on the PDF. "
                "Cite every factual claim using the provided "
                "source IDs, such as [S1]. "
                "Never invent source IDs.",
            ),
            (
                "human",
                "PDF CONTEXT:\n{context}\n\nQUESTION:\n{query}",
            ),
        ])

        self.answer_chain = (
            answer_prompt
            | self.answer_llm
            | StrOutputParser()
        )

        # Citation verification chain
        verification_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Verify whether each factual claim in the answer is supported
by its cited PDF evidence.

Use only the provided evidence.

Return valid JSON only, using this structure:

{{
    "claims": [
        {{
            "claim": "A factual claim",
            "citation_ids": ["S1"],
            "status": "SUPPORTED",
            "reason": "Explanation"
        }}
    ]
}}

Allowed statuses:
SUPPORTED
NOT_SUPPORTED
INSUFFICIENT_EVIDENCE

Do not use outside knowledge.
Do not invent citation IDs.""",
            ),
            (
                "human",
                "ANSWER:\n{answer}\n\nPDF EVIDENCE:\n{evidence}",
            ),
        ])

        self.verification_chain = (
            verification_prompt
            | self.verification_llm
            | StrOutputParser()
        )

    def verify_citations(self, answer: str, citations: list[dict]) -> dict:
        if not citations:
            return {"status": "NO_CITATIONS", "claims": []}

        evidence = "\n\n".join(
            f"[{item['source_id']}]\n"
            f"Page: {item['page_number']}\n"
            f"Content: {item['excerpt']}"
            for item in citations
        )

        valid_ids = {item["source_id"] for item in citations}

        try:
            content = self.verification_chain.invoke({
                "answer": answer,
                "evidence": evidence,
            })

            if not content:
                raise ValueError("Empty verification response")

            content = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                content.strip(),
                flags=re.IGNORECASE,
            )

            result = json.loads(content)
            claims = result.get("claims", [])

            if not isinstance(claims, list) or not claims:
                raise ValueError("No claims returned")

            allowed_statuses = {
                "SUPPORTED",
                "NOT_SUPPORTED",
                "INSUFFICIENT_EVIDENCE",
            }

            for claim in claims:
                if not isinstance(claim, dict):
                    raise ValueError("Invalid claim format")

                status = claim.get("status")
                citation_ids = claim.get("citation_ids")

                if status not in allowed_statuses:
                    raise ValueError("Invalid claim status")

                if not isinstance(citation_ids, list) or not citation_ids:
                    raise ValueError("Missing citation IDs")

                if not all(
                    isinstance(source_id, str) and source_id in valid_ids
                    for source_id in citation_ids
                ):
                    raise ValueError("Invalid citation ID")

            statuses = {claim["status"] for claim in claims}

            if "NOT_SUPPORTED" in statuses:
                final_status = "NEEDS_REVIEW"
            elif "INSUFFICIENT_EVIDENCE" in statuses:
                final_status = "INSUFFICIENT_EVIDENCE"
            else:
                final_status = "VERIFIED"

            return {
                "status": final_status,
                "claims": claims,
            }

        except Exception as exc:
            print(f"Citation verification failed: {exc}")
            return {
                "status": "UNVERIFIED",
                "claims": [],
            }

    def generate_answer(self, query: str, context_chunks: list[dict]) -> dict:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not context_chunks:
            return {
                "answer": "I don't know based on the provided PDF context.",
                "citations": [],
                "citation_verification": {
                    "status": "NOT_REQUIRED",
                    "claims": [],
                },
            }

        source_map = {}
        context_parts = []

        for index, chunk in enumerate(context_chunks, start=1):
            payload = chunk["payload"]
            source_id = f"S{index}"

            source = {
                "source_id": source_id,
                "filename": payload.get("filename", "Unknown"),
                "page_number": payload["page_number"],
                "excerpt": payload["text"],
            }

            source_map[source_id] = source

            context_parts.append(
                f"[{source_id}]\n"
                f"Filename: {source['filename']}\n"
                f"Page: {source['page_number']}\n"
                f"Content: {source['excerpt']}"
            )

        context = "\n\n".join(context_parts)

        raw_answer = self.answer_chain.invoke({
            "context": context,
            "query": query,
        })

        if not raw_answer:
            raise RuntimeError("Groq returned an empty response")

        referenced_ids = re.findall(
            r"(?:\[|【)(S\d+)(?:\]|】)",
            raw_answer,
        )

        cited_ids = list(dict.fromkeys(
            source_id
            for source_id in referenced_ids
            if source_id in source_map
        ))

        citations = [
            source_map[source_id]
            for source_id in cited_ids
        ]

        if raw_answer.strip().lower().startswith("i don't know"):
            verification = {
                "status": "NOT_REQUIRED",
                "claims": [],
            }
        else:
            verification = self.verify_citations(raw_answer, citations)

        # Preserve existing behaviour: citation details are
        # returned separately from the displayed answer.
        answer = re.sub(
            r"(?:\[|【)S\d+(?:\]|】)",
            "",
            raw_answer,
        ).strip()

        return {
            "answer": answer,
            "citations": citations,
            "citation_verification": verification,
        }
