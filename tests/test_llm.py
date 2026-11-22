
import pytest

from app.generation.llm import LLMGenerator


class FakeChain:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, input_data):
        return next(self.responses)


@pytest.fixture
def generator():
    instance = LLMGenerator.__new__(LLMGenerator)
    instance.answer_chain = None
    instance.verification_chain = None
    return instance


def sample_chunks():
    return [
        {
            "payload": {
                "filename": "research.pdf",
                "page_number": 3,
                "text": "Small fixtures help automated test suites run faster.",
            }
        },
        {
            "payload": {
                "filename": "research.pdf",
                "page_number": 4,
                "text": "Tests should verify expected application behavior.",
            }
        },
    ]


def supported_verification():
    return """
    {
        "claims": [
            {
                "claim": "Small fixtures improve test speed.",
                "citation_ids": ["S1"],
                "status": "SUPPORTED",
                "reason": "Supported by the provided PDF evidence."
            }
        ]
    }
    """


def test_valid_citation_mapping(generator):
    generator.answer_chain = FakeChain([
        "Small fixtures improve test speed [S1]."
    ])
    generator.verification_chain = FakeChain([
        supported_verification()
    ])

    result = generator.generate_answer(
        query="How can test suites run faster?",
        context_chunks=sample_chunks(),
    )

    assert result["answer"] == "Small fixtures improve test speed ."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_id"] == "S1"
    assert result["citations"][0]["filename"] == "research.pdf"
    assert result["citations"][0]["page_number"] == 3
    assert result["citation_verification"]["status"] == "VERIFIED"


def test_multiple_citations(generator):
    generator.answer_chain = FakeChain([
        "Fixtures improve speed [S1]. Tests verify behavior [S2]."
    ])

    generator.verification_chain = FakeChain([
        """
        {
            "claims": [
                {
                    "claim": "Fixtures improve speed.",
                    "citation_ids": ["S1"],
                    "status": "SUPPORTED",
                    "reason": "Evidence supports the claim."
                },
                {
                    "claim": "Tests verify behavior.",
                    "citation_ids": ["S2"],
                    "status": "SUPPORTED",
                    "reason": "Evidence supports the claim."
                }
            ]
        }
        """
    ])

    result = generator.generate_answer(
        query="Explain testing.",
        context_chunks=sample_chunks(),
    )

    assert len(result["citations"]) == 2
    assert [c["source_id"] for c in result["citations"]] == ["S1", "S2"]


def test_invalid_source_id_is_removed(generator):
    generator.answer_chain = FakeChain([
        "This claim uses an invalid source [S99]."
    ])

    result = generator.generate_answer(
        query="Explain testing.",
        context_chunks=sample_chunks(),
    )

    assert "[S99]" not in result["answer"]
    assert result["citations"] == []
    assert result["citation_verification"]["status"] == "NO_CITATIONS"


def test_duplicate_citations_are_deduplicated(generator):
    generator.answer_chain = FakeChain([
        "First claim [S1]. Another claim [S1]."
    ])
    generator.verification_chain = FakeChain([
        supported_verification()
    ])

    result = generator.generate_answer(
        query="Explain testing.",
        context_chunks=sample_chunks(),
    )

    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_id"] == "S1"


def test_empty_context_returns_no_citations(generator):
    result = generator.generate_answer(
        query="What is testing?",
        context_chunks=[],
    )

    assert result["answer"] == (
        "I don't know based on the provided PDF context."
    )
    assert result["citations"] == []
    assert result["citation_verification"]["status"] == "NOT_REQUIRED"


def test_empty_llm_response_raises_error(generator):
    generator.answer_chain = FakeChain([None])

    with pytest.raises(RuntimeError, match="Groq returned an empty response"):
        generator.generate_answer(
            query="Explain testing.",
            context_chunks=sample_chunks(),
        )
