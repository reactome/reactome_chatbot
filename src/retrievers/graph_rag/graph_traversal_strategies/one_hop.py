"""
One-hop graph traversal strategy.

This module implements a strategy that traverses one hop from seed nodes,
collecting neighbors with relationship type prioritization.
"""

from typing import List, Dict, Any
from collections import defaultdict

from .base import GraphTraversalStrategy
from ..types import IGraphClient


class OneHopStrategy(GraphTraversalStrategy):
    """
    One-hop traversal strategy that collects direct neighbors of seed nodes.
    
    This strategy prioritizes relationship types and limits the number of
    neighbors per type to maintain manageable context sizes.
    """
    
    name = "one_hop"
    max_neighbors_per_type: int = 5
    max_total: int = 10

    _order = [
        "PartOf", "SubPathwayOf", "HasInput", "HasOutput", "HasCatalyst", "HasComponent",
        "AssociatedWith", "Treats", "HasDiseaseVariant", "ActsOn", "Precedes",
    ]

    def traverse(
        self,
        graph_client: IGraphClient,
        seed_ids: List[str],  # Already correct - uses stable IDs
        cfg: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform one-hop traversal from seed nodes.

        Args:
            graph_client: The graph database client
            seed_ids: List of seed node IDs
            cfg: Configuration dictionary

        Returns:
            Dictionary mapping seed IDs to their neighbors organized by relationship type
        """
        per_type = int(cfg.get("max_neighbors_per_type", self.max_neighbors_per_type))
        total_cap = int(cfg.get("max_total", self.max_total))

        # Fetch seed node info using stable IDs
        seed_query = "MATCH (n) WHERE n.stId IN $ids RETURN n.stId AS node_id, n"
        seed_recs = graph_client.invoke(seed_query, {"ids": seed_ids})
        seeds: Dict[str, Dict[str, Any]] = {}
        
        for rec in seed_recs:
            n = rec["n"]
            nid = rec["node_id"]  # Now a string (stable_id)
            seeds[nid] = {
                "node_id": nid,
                "title": n.get("displayName") or n.get("name") or f"node:{nid}",
                "labels": list(n.labels),
                "preferred_id": n.get("preferred_id"),
                "description": n.get("description"),
                "id": n.get("id"),
                "associated_pathway": n.get("associated_pathway"),
                "regulated_by": n.get("regulated_by"),
                "url": n.get("url"),
                # Add any other seed props here
                **{k: v for k, v in dict(n).items() if k not in (
                    "displayName", "name", "preferred_id", "description", "id", 
                    "associated_pathway", "regulated_by", "url"
                )}
            }
            
        cypher = """
        UNWIND $seed_ids AS seed_id
        MATCH (seed) WHERE seed.stId = seed_id
        CALL {
          WITH seed, seed_id
          MATCH (seed)-[rel]-(nbr)
          WHERE seed <> nbr
          WITH seed_id, type(rel) AS rel_type, rel, nbr
          ORDER BY nbr.stId ASC, id(rel) ASC
          WITH seed_id, rel_type, collect({nbr: nbr, rel: rel})[0..$per_type] AS pairs
          UNWIND pairs AS pair
          RETURN
            rel_type,
            pair.nbr AS neighbor,
            pair.nbr.stId AS neighbor_id,
            pair.rel AS rel,
            startNode(pair.rel).stId AS rel_start_id,
            endNode(pair.rel).stId AS rel_end_id,
            coalesce(pair.rel.score, 0.0) AS rel_score,
            properties(pair.rel) AS rel_props
        }
        RETURN
          seed_id,
          rel_type,
          neighbor,
          neighbor_id,
          rel,
          rel_start_id,
          rel_end_id,
          rel_score,
          rel_props
        """

        records = graph_client.invoke(cypher, {"seed_ids": seed_ids, "per_type": per_type})

        rel_priority = {t: i for i, t in enumerate(self._order)}
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {sid: defaultdict(list) for sid in seed_ids}

        for rec in records:
            sid = rec["seed_id"]  # Now a string
            rel_type = rec["rel_type"]
            s_id = rec["rel_start_id"]  # Now a string
            e_id = rec["rel_end_id"]  # Now a string
            direction = "out" if s_id == sid else ("in" if e_id == sid else "undirected")
            neighbor = rec["neighbor"]

            grouped[sid][rel_type].append({
                "node_id": rec["neighbor_id"],  # Now a string
                "title": neighbor.get("displayName") or neighbor.get("name") or f"node:{rec['neighbor_id']}",
                "labels": list(neighbor.labels),
                "properties": dict(neighbor),
                "rel_type": rel_type,
                "rel_direction": direction,
                "rel_score": float(rec["rel_score"]),
                "rel_props": dict(rec["rel_props"]),
            })

        final_output: Dict[str, Dict[str, Any]] = {}

        for sid, rel_map in grouped.items():
            total = 0
            sorted_rels = sorted(rel_map.items(), key=lambda x: (rel_priority.get(x[0], len(self._order)), x[0]))
            neighbors_by_rel: Dict[str, List[Dict[str, Any]]] = {}
            
            for rel_type, neighbors in sorted_rels:
                if total >= total_cap:
                    break
                neighbors.sort(key=lambda x: (-x["rel_score"], x["node_id"]))
                limit = min(total_cap - total, len(neighbors))
                neighbors_by_rel[rel_type] = neighbors[:limit]
                total += limit
    
            final_output[sid] = {
                "seed": seeds.get(sid, {"node_id": sid}),  # always include the seed block
                "neighbors": neighbors_by_rel
            }
            
        return final_output


class OneHopRenderer:
    """
    Renderer for one-hop traversal results.
    
    Provides methods to convert one-hop traversal results to JSON or LLM-friendly text.
    """
    
    @staticmethod
    def to_json(data: Dict[str, Dict[str, Any]], pretty: bool = True) -> str:
        """
        Convert one-hop results to JSON format.
        
        Args:
            data: One-hop traversal results
            pretty: Whether to format with indentation
            
        Returns:
            JSON string representation
        """
        import json
        return json.dumps(data, indent=2 if pretty else None)

    @staticmethod
    def to_llm_text(data: Dict[str, Dict[str, Any]]) -> str:
        """
        Convert one-hop results to LLM-friendly text format.
        
        Args:
            data: One-hop traversal results
            
        Returns:
            Formatted text string for LLM consumption
        """
        lines = []
        for sid, info in data.items():
            seed = info.get("seed", {})
            seed_title = seed.get("title", f"node:{sid}")
            lines.append(f"## Node: {seed_title}")
            
            # Seed properties
            for prop in ("preferred_id", "description", "associated_pathway", "id", "regulated_by", "url"):
                val = seed.get(prop)
                if val is not None:
                    lines.append(f"* {prop}: {val}")
                    
            # Any other properties (skip name/displayName as those are used as title, and text_content)
            extra_props = [k for k in seed.keys() if k not in {
                "node_id", "title", "labels", "preferred_id", "description", 
                "associated_pathway", "id", "regulated_by", "url", "name", "displayName", "text_content"
            }]
            for k in extra_props:
                lines.append(f"* {k}: {seed[k]}")
                
            # Neighbors
            neighbors = info.get("neighbors", {})
            if not neighbors:
                lines.append("_No direct neighbors found._\n")
                continue
                
            for rel_type, nlist in neighbors.items():
                lines.append(f"\n### Relation: {rel_type}")
                for n in nlist:
                    n_title = n.get("title", f"node:{n.get('node_id', '?')}")
                    n_labels = ", ".join(n.get("labels", []))
                    props = []
                    for k, v in n.get("properties", {}).items():
                        if k not in {"displayName", "name", "text_content"} and v:
                            props.append(f"{k}: {v}")
                    url = n.get("properties", {}).get("url")
                    if url:
                        props.append(f"url: {url}")
                    prop_str = "; ".join(props)
                    lines.append(f"- **{n_title}** ({n_labels})" + (f" – {prop_str}" if prop_str else ""))
            lines.append("")
            
        return "\n".join(lines).strip()
