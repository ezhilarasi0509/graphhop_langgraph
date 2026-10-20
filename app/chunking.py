import re


def chunk_document(doc_id: str, text: str, max_words: int = 160, overlap_sents: int = 1) -> list[dict]:
    """Sentence-aware chunking. First line is the title -> used as the 'section path'.
    Chunk id = '<doc_id>:<n>' is deterministic, so re-ingestion hits the same ids (idempotent)."""
    title, _, body = text.strip().partition("\n")
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")) if s.strip()]
    chunks, cur, fresh = [], [], 0
    for s in sents:
        cur.append(s)
        fresh += 1
        if sum(len(x.split()) for x in cur) >= max_words:
            chunks.append(cur)
            cur = cur[-overlap_sents:] if overlap_sents else []
            fresh = 0
    if fresh:
        chunks.append(cur)
    return [{"id": f"{doc_id}:{i}", "doc_id": doc_id, "section": title.strip(),
             "text": f"{title.strip()}. " + " ".join(c)} for i, c in enumerate(chunks)]
