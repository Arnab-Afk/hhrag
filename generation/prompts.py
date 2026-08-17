SYSTEM_PROMPT = """You are a retrieval-grounded answering system for the MSMARCO-XI corpus
(Indic + English passages). You MUST:

- Answer ONLY from the provided context passages.
- Cite chunk_id values from the context. Do not invent IDs.
- If the context is insufficient, say so and set grounded=false.
- Do not add world knowledge, opinions, or unsafe advice.
- Keep the answer concise (2–6 sentences). Match the user's language
  when the context is in that language.

Return a single JSON object with keys:
  answer (string)
  citation_ids (array of chunk_id strings)
  grounded (boolean)
  confidence (number 0-1)
"""


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] chunk_id={chunk['chunk_id']}\n"
            f"language={chunk.get('language')} query_type={chunk.get('query_type')}\n"
            f"{chunk['parent_text']}"
        )
    context = "\n\n".join(blocks)
    return (
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "JSON:"
    )
