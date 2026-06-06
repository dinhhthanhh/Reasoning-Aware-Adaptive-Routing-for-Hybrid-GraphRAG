"""Knowledge graph builder from processed legal documents.

Loads processed documents, runs NER, extracts triples,
and batch-inserts into Neo4j. Produces Vietnamese legal
knowledge graph with typed relations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from tqdm import tqdm

from graph.neo4j_client import Neo4jClient
from ner.factory import get_ner_model
from ner.vi_ner import Entity


@dataclass
class Triple:
    """A knowledge graph triple (subject, predicate, object).

    Attributes:
        subject: Source entity text.
        subject_type: Source entity type.
        predicate: Relation name.
        object_: Target entity text.
        object_type: Target entity type.
        source_doc: Document ID this triple was extracted from.
    """

    subject: str
    subject_type: str
    predicate: str
    object_: str
    object_type: str
    source_doc: str = ""


@dataclass
class KGBuildReport:
    """Report from the knowledge graph build process.

    Attributes:
        documents_processed: Number of documents processed.
        entities_extracted: Total entities extracted by NER.
        triples_extracted: Total triples created.
        nodes_inserted: Nodes inserted into Neo4j.
        relations_inserted: Relations inserted into Neo4j.
        errors: List of error messages.
    """

    documents_processed: int = 0
    entities_extracted: int = 0
    triples_extracted: int = 0
    nodes_inserted: int = 0
    relations_inserted: int = 0
    errors: list[str] = field(default_factory=list)


# Relation patterns for Vietnamese legal text
ARTICLE_BELONGS_TO = re.compile(
    r"(Điều\s+\d+[a-zđ]?)\s*.*?(Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư)\s+[^\n,.]{3,60}",
    re.IGNORECASE,
)
ARTICLE_REFERENCES = re.compile(
    r"(Điều\s+\d+[a-zđ]?)\s*.*?(?:quy định tại|theo|căn cứ)\s+(Điều\s+\d+[a-zđ]?)",
    re.IGNORECASE,
)
SUBJECT_REGULATED_BY = re.compile(
    r"((?:tổ chức|cơ quan|doanh nghiệp|cá nhân|người)[^,.\n]{0,40})"
    r".*?(?:theo|được điều chỉnh bởi|tuân theo)\s+"
    r"((?:Bộ\s+luật|Luật|Nghị\s+định)[^,.\n]{3,60})",
    re.IGNORECASE,
)
ARTICLE_ABOUT = re.compile(
    r"(Điều\s+\d+[a-zđ]?)\.\s*([^\n]{5,80})",
    re.IGNORECASE,
)


def extract_triples(
    text: str,
    entities: list[Entity],
    doc_id: str,
) -> list[Triple]:
    """Extract knowledge graph triples from text and NER entities.

    Extracts four relation types:
    - (Điều X) -[THUỘC]-> (Luật Y)
    - (Điều X) -[THAM_CHIẾU]-> (Điều Z)
    - (Tổ chức A) -[ĐƯỢC_ĐIỀU_CHỈNH_BỞI]-> (Luật Y)
    - (Điều X) -[QUY_ĐỊNH_VỀ]-> (Khái niệm C)

    Args:
        text: Full document text.
        entities: NER entities from the text.
        doc_id: Source document identifier.

    Returns:
        List of extracted Triple objects.
    """
    triples: list[Triple] = []

    # Pattern 1: Article belongs to Law
    for m in ARTICLE_BELONGS_TO.finditer(text):
        article = m.group(1).strip()
        law = m.group(0).split(m.group(1))[-1].strip().rstrip(",.")
        if len(law) > 5:
            triples.append(Triple(
                subject=article,
                subject_type="LegalArticle",
                predicate="THUỘC",
                object_=law,
                object_type="Law",
                source_doc=doc_id,
            ))

    # Pattern 2: Article references another article
    for m in ARTICLE_REFERENCES.finditer(text):
        source_article = m.group(1).strip()
        target_article = m.group(2).strip()
        if source_article != target_article:
            triples.append(Triple(
                subject=source_article,
                subject_type="LegalArticle",
                predicate="THAM_CHIẾU",
                object_=target_article,
                object_type="LegalArticle",
                source_doc=doc_id,
            ))

    # Pattern 3: Subject regulated by law
    for m in SUBJECT_REGULATED_BY.finditer(text):
        subject = m.group(1).strip()
        law = m.group(2).strip()
        if len(subject) > 3 and len(law) > 5:
            triples.append(Triple(
                subject=subject,
                subject_type="LegalSubject",
                predicate="ĐƯỢC_ĐIỀU_CHỈNH_BỞI",
                object_=law,
                object_type="Law",
                source_doc=doc_id,
            ))

    # Pattern 4: Article about a concept (from article titles)
    for m in ARTICLE_ABOUT.finditer(text):
        article = m.group(1).strip()
        concept = m.group(2).strip().rstrip(".")
        if len(concept) > 3:
            triples.append(Triple(
                subject=article,
                subject_type="LegalArticle",
                predicate="QUY_ĐỊNH_VỀ",
                object_=concept,
                object_type="LegalConcept",
                source_doc=doc_id,
            ))

    # NER-based triples: link entities found together in the same document
    legal_entities = [e for e in entities if e.label == "LEGAL_TERM"]
    org_entities = [e for e in entities if e.label == "ORGANIZATION"]

    for org in org_entities:
        for legal in legal_entities:
            if abs(org.start - legal.start) < 500:  # proximity heuristic
                triples.append(Triple(
                    subject=org.text,
                    subject_type="Organization",
                    predicate="ĐƯỢC_ĐIỀU_CHỈNH_BỞI",
                    object_=legal.text,
                    object_type="LegalArticle",
                    source_doc=doc_id,
                ))

    return triples


def build_knowledge_graph(
    processed_dir: str | Path,
    neo4j_client: Neo4jClient,
    ner_model: Any,
    kg_export_dir: str | Path | None = None,
) -> KGBuildReport:
    """Build knowledge graph from processed legal documents.

    Pipeline:
    1. Load processed docs from JSON files
    2. Run NER batch extraction
    3. Extract triples using patterns + NER entities
    4. Batch insert nodes and relations into Neo4j
    5. Create indexes
    6. Export triples to JSON

    Args:
        processed_dir: Directory containing processed JSON docs.
        neo4j_client: Initialized Neo4j client.
        ner_model: Initialized Vietnamese NER model.
        kg_export_dir: Optional directory to export triples as JSON.

    Returns:
        KGBuildReport with processing statistics.
    """
    report = KGBuildReport()
    processed_path = Path(processed_dir)
    json_files = sorted(processed_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in {}", processed_path)
        return report

    logger.info("Building KG from {} documents", len(json_files))

    all_triples: list[Triple] = []
    all_nodes: dict[str, dict[str, Any]] = {}  # name -> node dict

    for json_file in tqdm(json_files, desc="Processing documents"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                doc = json.load(f)

            doc_id = doc.get("doc_id", json_file.stem)
            content = doc.get("content", "")
            title = doc.get("title", "")
            full_text = f"{title}\n{content}" if title else content

            if not full_text.strip():
                continue

            report.documents_processed += 1

            # NER extraction
            entities_list = ner_model.extract([full_text])
            entities = entities_list[0] if entities_list else []
            report.entities_extracted += len(entities)

            # Extract triples
            triples = extract_triples(full_text, entities, doc_id)
            all_triples.extend(triples)
            report.triples_extracted += len(triples)

            # Collect unique nodes
            for triple in triples:
                all_nodes[triple.subject] = {
                    "name": triple.subject,
                    "type": triple.subject_type,
                    "properties": {"source_doc": doc_id},
                }
                all_nodes[triple.object_] = {
                    "name": triple.object_,
                    "type": triple.object_type,
                    "properties": {"source_doc": doc_id},
                }

            # Also add NER entities as nodes
            for ent in entities:
                if ent.text.strip() and len(ent.text.strip()) > 1:
                    all_nodes[ent.text] = {
                        "name": ent.text,
                        "type": ent.label,
                        "properties": {"source_doc": doc_id, "confidence": ent.confidence},
                    }

        except (json.JSONDecodeError, IOError) as exc:
            error_msg = f"Failed to process {json_file}: {exc}"
            logger.warning(error_msg)
            report.errors.append(error_msg)

    # Insert into Neo4j
    if all_nodes:
        logger.info("Inserting {} unique nodes into Neo4j", len(all_nodes))
        report.nodes_inserted = neo4j_client.batch_insert_nodes(list(all_nodes.values()))

    if all_triples:
        logger.info("Inserting {} relations into Neo4j", len(all_triples))
        relations = [
            {
                "source": t.subject,
                "target": t.object_,
                "relation_type": t.predicate,
                "properties": {"source_doc": t.source_doc},
            }
            for t in all_triples
        ]
        report.relations_inserted = neo4j_client.batch_insert_relations(relations)

    # Create indexes
    neo4j_client.create_indexes()

    # Export triples to JSON
    if kg_export_dir:
        export_path = Path(kg_export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        triples_file = export_path / "triples.json"
        triples_data = [
            {
                "subject": t.subject,
                "subject_type": t.subject_type,
                "predicate": t.predicate,
                "object": t.object_,
                "object_type": t.object_type,
                "source_doc": t.source_doc,
            }
            for t in all_triples
        ]
        with open(triples_file, "w", encoding="utf-8") as f:
            json.dump(triples_data, f, ensure_ascii=False, indent=2)
        logger.info("Exported {} triples to {}", len(triples_data), triples_file)

    logger.info(
        "KG build complete | docs={} | entities={} | triples={} | nodes={} | relations={}",
        report.documents_processed,
        report.entities_extracted,
        report.triples_extracted,
        report.nodes_inserted,
        report.relations_inserted,
    )
    return report
