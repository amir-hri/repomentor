import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "qwen3:1.7b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5
INITIAL_CANDIDATES = 12
MAX_FACTS_PER_SOURCE = 2
MAX_TOTAL_FACTS = 6

CHUNKS_PATH = Path("data/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
OLLAMA_URL = "http://localhost:11434/api/chat"

STOPWORDS = {
    "what", "when", "where", "which", "who", "why", "how",
    "are", "does", "the", "this", "that", "with", "from",
    "into", "about", "handled",
}


def call_ollama(
    messages: list[dict],
    temperature: float = 0,
) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    expanded = []

    for token in tokens:
        expanded.append(token)
        expanded.extend(token.split("_"))

        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])

    return expanded


def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def find_related_identifiers(
    question: str,
    chunks: list[dict],
    candidate_indices: list[int],
) -> list[str]:
    """
    Find code identifiers in initially retrieved chunks that overlap
    with words from the user's question.
    """

    question_terms = {
        token
        for token in tokenize(question)
        if len(token) >= 4 and token not in STOPWORDS
    }

    identifier_counts = Counter()

    for index in candidate_indices:
        content = chunks[index]["content"]

        identifiers = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            content,
        )

        for identifier in identifiers:
            identifier_terms = set(tokenize(identifier))

            if "_" in identifier and question_terms & identifier_terms:
                identifier_counts[identifier] += 1

    return [
        identifier
        for identifier, _ in identifier_counts.most_common(25)
    ]


def is_test_file(file_path: str) -> bool:
    return (
        "/tests/" in file_path
        or file_path.startswith("tests/")
        or file_path.startswith("frontend/tests/")
    )


def create_citation(chunk: dict) -> str:
    return (
        f"[{chunk['file_path']}:"
        f"{chunk['start_line']}-{chunk['end_line']}]"
    )


def create_source_url(chunk: dict) -> str:
    encoded_path = quote(chunk["file_path"])

    return (
        f"https://gitlab.com/{chunk['project_path']}/-/blob/"
        f"{chunk['last_commit_id']}/{encoded_path}"
        f"#L{chunk['start_line']}-{chunk['end_line']}"
    )


def extract_facts(question: str, chunk: dict) -> list[str]:
    """
    Ask the model to extract facts from only one source.
    Python attaches the citation afterward.
    """

    result = call_ollama(
        [
            {
                "role": "system",
                "content": f"""
You are extracting facts from exactly one repository source.

Rules:
1. Use only the supplied source.
2. Extract at most {MAX_FACTS_PER_SOURCE} facts relevant to the question.
3. Output each fact as a bullet beginning with "- ".
4. Do not add citations; the program adds them automatically.
5. Do not use outside knowledge or make assumptions.
6. If the source does not directly help answer the question, output:
NONE
""",
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

File:
{chunk['file_path']}

Source:
{chunk['content']}
""",
            },
        ],
        temperature=0,
    )

    if result.strip().upper() == "NONE":
        return []

    citation = create_citation(chunk)
    facts = []

    for line in result.splitlines():
        line = line.strip()

        if not line.startswith(("- ", "* ")):
            continue

        fact = line[2:].strip()

        # Remove any citation the model added itself.
        fact = re.sub(r"\[[^\]]+\]\s*$", "", fact).strip()

        if fact:
            facts.append(f"- {fact} {citation}")

        if len(facts) == MAX_FACTS_PER_SOURCE:
            break

    return facts


# Load indexed data
chunks = load_chunks()
embeddings = np.load(EMBEDDINGS_PATH)

if len(chunks) != len(embeddings):
    raise RuntimeError(
        "Chunks and embeddings do not match. "
        "Run create_embeddings.py again."
    )

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

documents = [
    tokenize(f"{chunk['file_path']}\n{chunk['content']}")
    for chunk in chunks
]

bm25 = BM25Okapi(documents)

question = input("Ask RepoMentor: ").strip()

if not question:
    raise ValueError("The question cannot be empty.")

# First retrieval pass
question_embedding = embedding_model.encode(
    question,
    normalize_embeddings=True,
)

initial_semantic_scores = embeddings @ question_embedding
initial_keyword_scores = bm25.get_scores(tokenize(question))

semantic_ranking = np.argsort(initial_semantic_scores)[::-1]
keyword_ranking = np.argsort(initial_keyword_scores)[::-1]

initial_candidates = list(
    dict.fromkeys(
        semantic_ranking[:INITIAL_CANDIDATES].tolist()
        + keyword_ranking[:INITIAL_CANDIDATES].tolist()
    )
)

# Extract code identifiers from the initial results
related_identifiers = find_related_identifiers(
    question,
    chunks,
    initial_candidates,
)

print("\nRelated code identifiers:")

if related_identifiers:
    print(", ".join(related_identifiers))
else:
    print("None found")

# Second retrieval pass using discovered identifiers
expanded_query = f"{question} {' '.join(related_identifiers)}"

expanded_embedding = embedding_model.encode(
    expanded_query,
    normalize_embeddings=True,
)

semantic_scores = np.maximum(
    initial_semantic_scores,
    embeddings @ expanded_embedding,
)

keyword_scores = bm25.get_scores(tokenize(expanded_query))

semantic_ranking = np.argsort(semantic_scores)[::-1]
keyword_ranking = np.argsort(keyword_scores)[::-1]

# Reciprocal Rank Fusion
combined_scores = np.zeros(len(chunks))

for ranking in (semantic_ranking, keyword_ranking):
    for rank, index in enumerate(ranking, start=1):
        combined_scores[index] += 1 / (60 + rank)

# Prefer implementation files unless the user asks about tests
if "test" not in question.lower():
    for index, chunk in enumerate(chunks):
        if is_test_file(chunk["file_path"]):
            combined_scores[index] *= 0.75

# Select different files for broader evidence
ranked_indices = np.argsort(combined_scores)[::-1]

best_indices = []
seen_files = set()

for index in ranked_indices:
    file_path = chunks[index]["file_path"]

    if file_path in seen_files:
        continue

    best_indices.append(index)
    seen_files.add(file_path)

    if len(best_indices) == TOP_K:
        break

retrieved_chunks = [chunks[index] for index in best_indices]

# Extract grounded facts one source at a time
answer_facts = []
used_chunks = []

for chunk in retrieved_chunks:
    facts = extract_facts(question, chunk)

    if facts:
        answer_facts.extend(facts)
        used_chunks.append(chunk)

    if len(answer_facts) >= MAX_TOTAL_FACTS:
        break

answer_facts = answer_facts[:MAX_TOTAL_FACTS]

print("\nAnswer:\n")

if answer_facts:
    for fact in answer_facts:
        print(fact)
else:
    print(
        "- I couldn't find enough evidence in the "
        "retrieved repository sources."
    )

print("\nSources:\n")

for chunk in used_chunks:
    print(create_citation(chunk))
    print(create_source_url(chunk))
    print()