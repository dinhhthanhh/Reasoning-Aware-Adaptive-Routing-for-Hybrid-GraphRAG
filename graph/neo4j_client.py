"""Neo4j client for Knowledge Graph operations.

Provides an interface for connecting to Neo4j, performing Cypher queries,
and managing batch document ingestion with memory optimization.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from neo4j import GraphDatabase, Session


NO_GRAPH_CONTEXT = "No relevant information found in the knowledge graph."


class Neo4jClient:
    """Client for interacting with Neo4j Knowledge Graph.
    
    Handles connection management, indexing, and batch ingestion
    using managed transactions to minimize memory usage.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Neo4j client.

        Args:
            config: Neo4j config dict with uri, user, password, and database.
        """
        self.uri = config.get("uri", "bolt://localhost:7687")
        self.user = config.get("user", "neo4j")
        self.password = config.get("password", "password")
        self.database = config.get("database", "neo4j")
        self.batch_size = config.get("batch_size", 500)

        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        logger.info("Neo4jClient initialized | uri={} | database={}", self.uri, self.database)

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    def verify_connection(self) -> bool:
        """Verify the connection to Neo4j.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("Neo4j connection verification failed: {}", e)
            return False

    def _get_driver(self):
        """Return the active driver."""
        return self._driver

    def batch_insert_nodes(self, nodes: list[dict[str, Any]]) -> int:
        """Insert nodes in batches using separate sessions per chunk.
        
        Args:
            nodes: List of node dictionaries.

        Returns:
            Total number of nodes inserted/merged.
        """
        total = 0
        for i in range(0, len(nodes), self.batch_size):
            chunk = nodes[i : i + self.batch_size]
            with self._get_driver().session(database=self.database) as session:
                count = session.execute_write(self._insert_nodes_tx, chunk)
                total += count
        
        logger.info("Batch inserted/merged {} nodes", total)
        return total

    @staticmethod
    def _insert_nodes_tx(tx, nodes: list[dict[str, Any]]) -> int:
        query = """
        UNWIND $nodes AS node
        MERGE (n:Entity {name: node.name})
        SET n.type = node.type,
            n += node.properties
        RETURN count(n) AS cnt
        """
        result = tx.run(query, nodes=nodes)
        record = result.single()
        return record["cnt"] if record else 0

    def batch_insert_relations(self, relations: list[dict[str, Any]]) -> int:
        """Insert relations in batches using separate sessions per chunk.

        Args:
            relations: List of relation dictionaries.

        Returns:
            Total number of relations inserted/merged.
        """
        total = 0
        for i in range(0, len(relations), self.batch_size):
            chunk = relations[i : i + self.batch_size]
            with self._get_driver().session(database=self.database) as session:
                try:
                    count = session.execute_write(self._insert_relations_tx, chunk)
                except Exception as e:
                    logger.warning("APOC relation insert failed, trying fallback: {}", e)
                    count = session.execute_write(self._insert_relations_no_apoc_tx, chunk)
                total += count

        logger.info("Batch inserted/merged {} relations", total)
        return total

    @staticmethod
    def _insert_relations_tx(tx, relations: list[dict[str, Any]]) -> int:
        query = """
        UNWIND $rels AS rel
        MATCH (a:Entity {name: rel.source})
        MATCH (b:Entity {name: rel.target})
        CALL apoc.merge.relationship(a, rel.relation_type, {}, rel.properties, b) YIELD rel AS r
        RETURN count(r) AS cnt
        """
        result = tx.run(query, rels=relations)
        record = result.single()
        return record["cnt"] if record else 0

    @staticmethod
    def _insert_relations_no_apoc_tx(tx, relations: list[dict[str, Any]]) -> int:
        from collections import defaultdict
        by_type = defaultdict(list)
        for rel in relations:
            by_type[rel["relation_type"]].append(rel)

        total = 0
        for rel_type, rels in by_type.items():
            safe_type = rel_type.replace(" ", "_").replace("-", "_")
            query = f"""
            UNWIND $rels AS rel
            MATCH (a:Entity {{name: rel.source}})
            MATCH (b:Entity {{name: rel.target}})
            MERGE (a)-[r:{safe_type}]->(b)
            SET r += rel.properties
            RETURN count(r) AS cnt
            """
            result = tx.run(query, rels=rels)
            record = result.single()
            total += record["cnt"] if record else 0
        return total

    def create_indexes(self) -> None:
        """Create indexes for efficient graph queries."""
        with self._get_driver().session(database=self.database) as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read Cypher query.

        Args:
            cypher: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """
        with self._get_driver().session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def get_multi_hop_context(self, query: str, top_k: int = 3) -> str:
        """Perform a multi-hop search to build reasoning context.

        Args:
            query: The natural language query.
            top_k: Number of start nodes to explore.

        Returns:
            Formatted context string.
        """
        keywords = self._extract_query_terms(query)
        if not keywords:
            return NO_GRAPH_CONTEXT

        try:
            starts = self._find_start_nodes(keywords, top_k=top_k)
            if not starts:
                return NO_GRAPH_CONTEXT

            context_parts: list[str] = []
            seen_paths: set[str] = set()

            for start in starts:
                start_name = start.get("name") or start.get("doc_id") or start.get("title") or "Unknown"
                start_type = start.get("type") or ",".join(start.get("labels", [])) or "Node"
                start_text = start.get("summary") or start.get("content") or start.get("title") or ""
                context_parts.append(
                    f"[Start: {start_name}] ({start_type})\n{str(start_text)[:1800]}"
                )

                paths = self._expand_paths(start["element_id"], max_paths=max(top_k * 2, 3))
                for path in paths:
                    path_key = " -> ".join(path.get("node_names", [])) + "|" + " -> ".join(path.get("rel_types", []))
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)

                    rel_chain = " -> ".join(path.get("rel_types", [])) or "RELATED"
                    node_chain = " -> ".join(path.get("node_names", []))
                    end_text = path.get("end_summary") or path.get("end_text") or ""
                    context_parts.append(
                        f"Path: {node_chain}\nRelation chain: {rel_chain}\nEvidence: {str(end_text)[:1200]}"
                    )

            return "\n\n".join(context_parts[: max(1, top_k * 5)])

        except Exception as exc:
            logger.warning("Neo4j multi-hop retrieval failed: {}", exc)
            return NO_GRAPH_CONTEXT

    @staticmethod
    def _extract_query_terms(query: str, max_terms: int = 12) -> list[dict[str, Any]]:
        """Extract robust search terms for Cypher substring matching."""
        stopwords = {
            "là", "gì", "của", "và", "hay", "hoặc", "theo", "được", "trong",
            "như", "nào", "bao", "nhiêu", "có", "không", "cho", "biết",
            "quy", "định", "quy định", "pháp", "luật", "pháp luật", "việt",
            "nam", "việt nam", "điều", "khoản", "văn", "bản", "trường", "hợp",
            "so", "sánh", "về", "với", "tại", "này", "đó",
            "một", "những", "các", "khi", "thì", "sẽ", "đã", "nếu", "liệu",
            "trước", "sau", "ngày", "tháng", "năm", "số", "hiện", "hành",
            "thông", "tư", "nghị", "nđ", "cp", "vbhn", "qđ", "ubnd",
            "what", "which", "who", "where", "when", "how", "the", "and",
            "or", "of", "in", "on", "to", "is", "are", "was", "were",
        }
        short_legal_tokens = {"cơ", "xử", "vi", "ly", "án", "hộ"}
        priority_phrases = {
            "luật đường bộ": 8,
            "đường bộ": 7,
            "đất dành cho giao thông": 8,
            "giao thông trong đô thị": 7,
            "tỷ lệ đất dành": 7,
            "đường sắt": 7,
            "đường hàng không": 7,
            "hàng không": 6,
            "sân bay": 6,
            "cảng hàng không": 7,
            "vùng trời sân bay": 7,
            "hoạt động bay": 6,
            "quy chế bay": 6,
            "phương thức bay": 6,
            "chất thải cồng kềnh": 8,
            "bao bì tái chế": 7,
            "tái chế bắt buộc": 7,
            "chất thải rắn": 6,
            "hội đồng nhân dân": 6,
            "ủy ban nhân dân": 6,
            "sỹ quan kiểm tra tàu biển": 8,
            "tàu biển": 6,
            "thừa phát lại": 6,
            "điều hành chuyến bay": 6,
            "cấp phép bay": 5,
            "chuyến bay": 4,
            "vốn oda": 6,
            "vốn vay": 5,
            "vốn vay ưu đãi": 6,
            "quy định cũ": 5,
            "điều khoản chuyển tiếp": 6,
            "xử lý hồ sơ": 4,
            "kết hôn": 5,
            "ly hôn": 5,
            "điều kiện kết hôn": 6,
            "thẩm quyền xử phạt": 6,
            "xử phạt vi phạm": 5,
            "cục hóa chất": 8,
            "hóa chất": 6,
            "quỹ đổi mới công nghệ quốc gia": 8,
            "đổi mới sáng tạo": 6,
            "quản lý y dược cổ truyền": 8,
            "y dược cổ truyền": 7,
            "thủ tục hành chính": 6,
            "phân cấp": 5,
            "con dấu": 7,
            "ký và sử dụng con dấu": 8,
            "giấy phép hoạt động": 7,
            "điều chỉnh giấy phép": 7,
            "bệnh viện tư nhân": 7,
            "quy mô giường bệnh": 7,
            "mỹ phẩm": 6,
            "bãi bỏ": 5,
            "hết hiệu lực": 5,
            "chuyển tiếp": 5,
            "hồ sơ": 5,
            "miễn giảm tiền thuê đất": 8,
            "miễn giảm tiền sử dụng đất": 8,
            "miễn giảm tiền": 6,
            "tiền thuê đất": 7,
            "dân tộc thiểu số": 7,
            "chuyển nhượng vốn": 7,
            "hỗ trợ lãi suất": 7,
            "lãi suất vay": 6,
        }
        query_lower = query.lower()
        token_sequence: list[str] = []
        unique_tokens: list[str] = []
        for term in re.findall(r"[\wÀ-ỹĐđ]+", query_lower, flags=re.UNICODE):
            if (len(term) < 3 and term not in short_legal_tokens) or term in stopwords:
                continue
            token_sequence.append(term)
            if term not in unique_tokens:
                unique_tokens.append(term)

        phrases: list[str] = []
        for n in (3, 2):
            for i in range(0, max(0, len(token_sequence) - n + 1)):
                phrase = " ".join(token_sequence[i : i + n])
                if phrase not in phrases:
                    phrases.append(phrase)

        weighted_terms: list[dict[str, Any]] = []
        legal_ref_patterns = (
            r"\b\d{1,5}/\d{4}/[A-ZÀ-ỸĐ0-9-]+",
            r"(?<!/)\b\d{1,5}/(?=[A-ZÀ-ỸĐ0-9-]*[A-ZÀ-ỸĐ])[A-ZÀ-ỸĐ0-9-]+",
        )
        for pattern in legal_ref_patterns:
            for match in re.finditer(pattern, query, flags=re.IGNORECASE):
                ref = match.group(0).lower()
                if ref not in {term["text"] for term in weighted_terms}:
                    weighted_terms.append({"text": ref, "weight": 10})
                normalized_ref = ref.replace("/", " ").replace("-", " ")
                if normalized_ref and normalized_ref not in {term["text"] for term in weighted_terms}:
                    weighted_terms.append({"text": normalized_ref, "weight": 9})

        for phrase, weight in priority_phrases.items():
            if phrase in query_lower:
                weighted_terms.append({"text": phrase, "weight": weight})

        # Sliding n-grams are useful, but too many generic n-grams can dominate
        # full-text search and pull the graph to unrelated legal domains.
        domain_tokens = {
            "đường", "bộ", "sắt", "hàng", "không", "bay", "sân", "cảng",
            "chất", "thải", "tái", "chế", "bao", "bì", "thuế", "đất",
            "hội", "đồng", "nhân", "dân", "ủy", "ban", "tàu", "biển",
            "quỹ", "kinh", "phí", "giấy", "phép", "hành", "nghề",
            "thẩm", "quyền", "trách", "nhiệm", "hiệu", "lực", "hồ", "sơ",
            "thủ", "tục", "bãi", "bỏ", "sửa", "đổi", "bổ", "sung", "chuyển",
            "tiếp", "công", "bố", "phân", "cấp", "con", "dấu", "ký", "sử",
            "dụng", "cục", "hóa", "mỹ", "phẩm", "lãi", "suất",
        }
        generic_phrase_heads = {
            "khi", "nếu", "phục", "đáp", "ứng", "thực", "hiện", "xác",
            "liệu", "giai", "đoạn", "quyết",
        }
        selected_phrases = []
        for phrase in phrases:
            tokens = phrase.split()
            if tokens and tokens[0] in generic_phrase_heads:
                continue
            if not any(token in domain_tokens for token in tokens):
                continue
            selected_phrases.append(phrase)

        for phrase in selected_phrases[:8]:
            if phrase not in {term["text"] for term in weighted_terms}:
                weighted_terms.append({"text": phrase, "weight": 4 if len(phrase.split()) == 3 else 3})
        for token in unique_tokens:
            if token not in {term["text"] for term in weighted_terms}:
                weighted_terms.append({"text": token, "weight": 1})

        weighted_terms = sorted(
            enumerate(weighted_terms),
            key=lambda item: (
                -float(item[1].get("weight", 1.0)),
                -len(str(item[1].get("text", "")).split()),
                item[0],
            ),
        )
        return [term for _, term in weighted_terms[:max_terms]]

    def _find_start_nodes(self, keywords: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Find candidate start nodes without scanning large content properties."""
        keyword_texts = self._select_fulltext_terms(keywords)
        search_query = self._build_fulltext_query(keyword_texts)
        candidates: list[dict[str, Any]] = []

        if search_query:
            for index_name, limit_multiplier in (
                ("legal_article_fulltext", 3),
                ("vector_chunk_fulltext", 2),
            ):
                try:
                    candidates.extend(
                        self._query_fulltext_index(
                            index_name=index_name,
                            search_query=search_query,
                            limit=max(top_k * limit_multiplier, top_k),
                        )
                    )
                except Exception as exc:
                    logger.debug("Fulltext lookup skipped for {}: {}", index_name, exc)

        candidates.extend(
            self._query_legal_doc_metadata(
                keywords=keywords,
                limit=max(top_k * 4, top_k),
            )
        )

        by_id: dict[str, dict[str, Any]] = {}
        for row in candidates:
            element_id = row.get("element_id")
            if not element_id:
                continue
            existing = by_id.get(element_id)
            if existing is None or row.get("score", 0) > existing.get("score", 0):
                by_id[element_id] = row

        label_boost = {"LegalArticle": 8.0, "VectorChunk": 1.5, "LegalDoc": 2.0}
        contradictory_domains = (
            ("đường bộ", "đường sắt"),
            ("đường sắt", "đường bộ"),
            ("hàng không", "đường bộ"),
            ("hàng không", "đường sắt"),
            ("chất thải", "hàng không"),
        )

        def rank(row: dict[str, Any]) -> tuple[float, float, int]:
            labels = row.get("labels", [])
            boost = max((label_boost.get(label, 0.0) for label in labels), default=0.0)
            haystack = " ".join(
                str(row.get(key, ""))
                for key in ("name", "title", "summary", "content", "doc_id", "article_id")
            ).lower()
            keyword_match = sum(
                float(kw.get("weight", 1.0))
                for kw in keywords
                if kw.get("text") and str(kw["text"]).lower() in haystack
            )
            anchor_match = sum(
                1.0
                for kw in keywords
                if float(kw.get("weight", 1.0)) >= 5.0
                and kw.get("text")
                and str(kw["text"]).lower() in haystack
            )
            query_terms = {str(kw.get("text", "")).lower() for kw in keywords}
            contradiction_penalty = sum(
                10.0
                for expected, unexpected in contradictory_domains
                if expected in query_terms and unexpected in haystack
            )
            return (
                keyword_match * 4.0
                + anchor_match * 8.0
                + float(row.get("score", 0.0))
                + boost
                - contradiction_penalty,
                float(row.get("degree", 0.0)),
                -len(row.get("name", "")),
            )

        ranked = sorted(by_id.values(), key=rank, reverse=True)
        filtered = [row for row in ranked if self._candidate_has_minimum_signal(row, keywords)]
        return (filtered or ranked)[:top_k]

    @staticmethod
    def _select_fulltext_terms(keywords: list[dict[str, Any]], max_terms: int = 8) -> list[str]:
        """Keep fulltext search anchored on phrases instead of generic tokens."""
        selected: list[str] = []
        fallback: list[str] = []
        generic_tokens = {
            "hiệu", "lực", "trách", "nhiệm", "thẩm", "quyền", "quyết", "hành",
            "chính", "pháp", "luật", "ngày", "tháng", "năm", "trước", "sau",
        }
        for kw in keywords:
            text = str(kw.get("text", "")).strip()
            if not text:
                continue
            weight = float(kw.get("weight", 1.0))
            is_phrase = " " in text
            if weight >= 3.0 or is_phrase:
                selected.append(text)
            elif len(text) >= 4 and text not in generic_tokens:
                fallback.append(text)

        for text in fallback:
            if len(selected) >= max_terms:
                break
            if text not in selected:
                selected.append(text)
        return selected[:max_terms]

    @staticmethod
    def _candidate_has_minimum_signal(row: dict[str, Any], keywords: list[dict[str, Any]]) -> bool:
        """Reject weak candidates pulled in by broad OR fulltext matches."""
        haystack = " ".join(
            str(row.get(key, ""))
            for key in ("name", "title", "summary", "content", "doc_id", "article_id")
        ).lower()
        weighted_match = sum(
            float(kw.get("weight", 1.0))
            for kw in keywords
            if kw.get("text") and str(kw["text"]).lower() in haystack
        )
        if weighted_match >= 5.0:
            return True
        return float(row.get("score", 0.0)) >= 8.0 and weighted_match >= 3.0

    @staticmethod
    def _build_fulltext_query(keyword_texts: list[str], max_terms: int = 8) -> str:
        """Build a conservative Lucene query for Neo4j fulltext indexes."""
        terms = []
        for text in keyword_texts[:max_terms]:
            cleaned = str(text)
            for char in ['\\', '+', '-', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '/']:
                cleaned = cleaned.replace(char, f"\\{char}")
            cleaned = cleaned.strip()
            if cleaned:
                terms.append(f'"{cleaned}"')
        return " OR ".join(terms)

    def _query_fulltext_index(self, index_name: str, search_query: str, limit: int) -> list[dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $search_query, {limit: $limit})
        YIELD node, score
        WITH node, score, labels(node) AS labels, properties(node) AS props
        OPTIONAL MATCH (node)-[r]-()
        WITH node, score, labels, props, count(r) AS degree,
             coalesce(props.name, props.title, props.article_id, props.doc_id, props.chunk_id, "") AS display,
             coalesce(props.title, "") AS title,
             coalesce(props.doc_id, props.law_id, "") AS doc_id,
             coalesce(props.article_id, "") AS article_id,
             coalesce(props.chunk_id, "") AS chunk_id,
             coalesce(props.content, props.text, props.content_preview, props.source_doc, "") AS content,
             coalesce(props.authority, "") AS authority,
             coalesce(props.type, "") AS node_type,
             coalesce(props.source, "") AS source,
             coalesce(props.issue_date, "") AS issue_date
        WITH node, score, degree, labels, display, title, doc_id, article_id,
             chunk_id, content, authority, node_type, source, issue_date,
             CASE
                WHEN "LegalArticle" IN labels THEN
                    "Title: " + title + "\nArticle ID: " + article_id +
                    "\nLaw ID: " + doc_id + "\nContent: " + content
                WHEN "VectorChunk" IN labels THEN
                    "Title: " + title + "\nDoc ID: " + doc_id +
                    "\nVector chunk: " + chunk_id + "\nContent: " + content
                ELSE content
             END AS summary
        RETURN elementId(node) AS element_id,
               display AS name,
               title,
               doc_id,
               article_id,
               chunk_id,
               content,
               summary,
               labels,
               node_type AS type,
               "" AS source_doc,
               authority,
               issue_date,
               source,
               score,
               degree
        """
        return self.query(cypher, {
            "index_name": index_name,
            "search_query": search_query,
            "limit": limit,
        })

    def _query_legal_doc_metadata(self, keywords: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        cypher = """
        MATCH (node:LegalDoc)
        WITH node, labels(node) AS labels, properties(node) AS props
        WITH node, labels, props,
             coalesce(props.title, props.doc_id, "") AS display,
             coalesce(props.title, "") AS title,
             coalesce(props.doc_id, "") AS doc_id,
             coalesce(props.authority, "") AS authority,
             coalesce(props.type, "") AS node_type,
             coalesce(props.source, "") AS source,
             coalesce(props.issue_date, "") AS issue_date
        WITH node, labels, display, title, doc_id, authority, node_type, source, issue_date,
             display + " " + title + " " + doc_id + " " + authority + " " +
             node_type + " " + source + " " + issue_date AS search_text
        WITH node, labels, display, title, doc_id, authority, node_type, source, issue_date,
             reduce(score = 0, kw IN $keywords |
                score + CASE
                    WHEN toLower(search_text) CONTAINS kw.text THEN kw.weight
                    ELSE 0
                END
             ) AS score
        WHERE score >= 3
        OPTIONAL MATCH (node)-[r]-()
        WITH node, labels, display, title, doc_id, authority, node_type, source, issue_date,
             score, count(r) AS degree,
             "Title: " + title + "\nDoc ID: " + doc_id +
             "\nType: " + node_type + "\nAuthority: " + authority +
             "\nIssue date: " + issue_date + "\nSource: " + source AS summary
        RETURN elementId(node) AS element_id,
               display AS name,
               title,
               doc_id,
               "" AS article_id,
               "" AS chunk_id,
               "" AS content,
               summary,
               labels,
               node_type AS type,
               "" AS source_doc,
               authority,
               issue_date,
               source,
               score,
               degree
        ORDER BY score DESC, degree DESC, size(display) ASC
        LIMIT $limit
        """
        return self.query(cypher, {"keywords": keywords, "limit": limit})

    def _expand_paths(self, start_element_id: str, max_paths: int) -> list[dict[str, Any]]:
        """Expand 1-2 hop paths from a start node and format path metadata."""
        cypher = """
        MATCH (start)
        WHERE elementId(start) = $start_id
        MATCH p = (start)-[*1..2]-(end)
        WHERE start <> end AND NOT end:VectorChunk
        WITH p, end, relationships(p) AS rels, nodes(p) AS ns, properties(end) AS end_props
        RETURN
            [node IN ns | coalesce(properties(node).name, properties(node).title, properties(node).article_id, properties(node).doc_id, properties(node).chunk_id, "Unknown")] AS node_names,
            [rel IN rels | type(rel) + coalesce(" (" + properties(rel).type + ")", "")] AS rel_types,
            coalesce(end_props.content, end_props.text, end_props.content_preview, end_props.title, end_props.source_doc, "") AS end_text,
            CASE
                WHEN "LegalDoc" IN labels(end) THEN
                    "Title: " + coalesce(end_props.title, "") +
                    "\nDoc ID: " + coalesce(end_props.doc_id, "") +
                    "\nType: " + coalesce(end_props.type, "") +
                    "\nAuthority: " + coalesce(end_props.authority, "") +
                    "\nIssue date: " + coalesce(end_props.issue_date, "") +
                    "\nSource: " + coalesce(end_props.source, "")
                WHEN "LegalArticle" IN labels(end) THEN
                    "Title: " + coalesce(end_props.title, "") +
                    "\nArticle ID: " + coalesce(end_props.article_id, "") +
                    "\nLaw ID: " + coalesce(end_props.law_id, "") +
                    "\nContent: " + coalesce(end_props.content, end_props.content_preview, "")
                WHEN "VectorChunk" IN labels(end) THEN
                    "Title: " + coalesce(end_props.title, "") +
                    "\nDoc ID: " + coalesce(end_props.doc_id, "") +
                    "\nVector chunk: " + coalesce(end_props.chunk_id, "") +
                    "\nContent: " + coalesce(end_props.content_preview, "")
                ELSE coalesce(end_props.content, end_props.text, end_props.content_preview, end_props.source_doc, "")
            END AS end_summary,
            length(p) AS hops
        ORDER BY hops ASC,
                 CASE
                    WHEN "LegalArticle" IN labels(end) THEN 0
                    WHEN "LegalDoc" IN labels(end) THEN 1
                    ELSE 2
                 END ASC
        LIMIT $max_paths
        """
        return self.query(cypher, {"start_id": start_element_id, "max_paths": max_paths})
