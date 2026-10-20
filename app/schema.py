"""The ontology. Constrained on purpose: an open-ended 'extract everything' prompt gives a graph nobody can query.
Relationships are written as (source)-[TYPE]->(target)."""
from typing import Literal
from pydantic import BaseModel, Field

ENTITY_TYPES = ["Company", "Person", "Product", "Location", "Regulation"]

# type -> (verb phrase used to turn an edge into a sentence, direction hint shown to the extractor)
REL_TYPES = {
    "SUPPLIES": ("supplies", "Company -> Company/Product being supplied"),
    "ACQUIRED": ("acquired", "acquirer Company -> acquired Company"),
    "COMPETES_WITH": ("competes with", "Company -> Company"),
    "FOUNDED_BY": ("was founded by", "Company -> Person"),
    "CEO_OF": ("is the CEO of", "Person -> Company"),
    "HEADQUARTERED_IN": ("is headquartered in", "Company -> Location"),
    "PRODUCES": ("produces", "Company -> Product"),
    "USES": ("uses", "Product/Company -> Product"),
    "PARTNERS_WITH": ("partners with", "Company -> Company"),
    "INVESTED_IN": ("invested in", "Company -> Company"),
    "SUBJECT_TO": ("is subject to", "Company -> Regulation"),
}
REL_UNION = "|".join(REL_TYPES)  # safe to interpolate into Cypher: a constant, never user input

EntityType = Literal[tuple(ENTITY_TYPES)]
RelType = Literal[tuple(REL_TYPES)]


class Rel(BaseModel):
    source: str = Field(min_length=1)
    source_type: EntityType
    relation: RelType
    target: str = Field(min_length=1)
    target_type: EntityType
    confidence: float = Field(ge=0, le=1)
    evidence: str = ""


class Extraction(BaseModel):
    relations: list[Rel]
