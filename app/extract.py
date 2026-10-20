"""Phase 1a: LLM extraction with a strict schema, validation, and repair-retry."""
import json
from pydantic import ValidationError
from . import config, llm
from .schema import ENTITY_TYPES, REL_TYPES, Extraction, Rel

_rels = "\n".join(f"- {k}: {hint}" for k, (_, hint) in REL_TYPES.items())
SYSTEM = f"""You extract a knowledge graph from text. Return ONLY JSON:
{{"relations": [{{"source": str, "source_type": str, "relation": str, "target": str,
"target_type": str, "confidence": number 0-1, "evidence": str}}]}}

Entity types (use exactly): {ENTITY_TYPES}
Relation types (use exactly), with direction source -> target:
{_rels}

Rules:
- Use the FULLEST canonical name seen in the text (e.g. "Helios Energy Corporation", not "Helios" or "the company").
- Resolve pronouns and "the company" to the actual entity. Never output pronouns as entities.
- Only extract relations explicitly stated. If unsure, skip. Do not invent types.
- "evidence" = a short quote (<25 words) copied from the text that justifies the relation.
- No relations found -> {{"relations": []}}"""


def extract(chunk_text: str) -> list[Rel]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": chunk_text}]
    for _ in range(3):
        raw = llm.chat(messages, json_mode=True, use_cache=True)  # cached by request hash
        try:
            rels = Extraction.model_validate(llm.parse_json(raw)).relations
            return [r for r in rels if r.confidence >= config.MIN_CONFIDENCE]
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid output: {str(e)[:500]}\nReturn corrected JSON only."}]
    print("  ! extraction failed after 3 attempts, skipping chunk")
    return []
