"""
Steiner tree graph traversal strategy.

This module implements a strategy that finds minimum-weight Steiner trees
connecting seed nodes using Neo4j's GDS library.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict, deque
import uuid

from .base import GraphTraversalStrategy
from ..types import IGraphClient


class SteinerTreeStrategy(GraphTraversalStrategy):
    """
    Steiner tree traversal strategy using Neo4j GDS.
    
    This strategy finds minimum-weight Steiner trees connecting seed nodes,
    which is useful for finding optimal paths between multiple seed nodes.
    """
    
    name = "steiner_tree"

    def _stable_ids_to_internal_ids(self, graph_client: IGraphClient, stable_ids: List[str]) -> Dict[str, int]:
        """Convert stable IDs to internal Neo4j IDs for GDS operations."""
        if not stable_ids:
            return {}
        
        query = """
        UNWIND $stable_ids as stable_id
        MATCH (n {stId: stable_id})
        RETURN stable_id, id(n) as internal_id
        """
        
        results = graph_client.invoke(query, {"stable_ids": stable_ids})
        return {record["stable_id"]: record["internal_id"] for record in results}

    def _internal_ids_to_stable_ids(self, graph_client: IGraphClient, internal_ids: List[int]) -> Dict[int, str]:
        """Convert internal Neo4j IDs back to stable IDs."""
        if not internal_ids:
            return {}
        
        query = """
        UNWIND $internal_ids as internal_id
        MATCH (n)
        WHERE id(n) = internal_id
        RETURN id(n) as internal_id, n.stId as stable_id
        """
        
        results = graph_client.invoke(query, {"internal_ids": internal_ids})
        return {record["internal_id"]: record["stable_id"] for record in results}

    def traverse(
        self,
        graph_client: IGraphClient,
        seed_ids: List[str],  # Changed from List[int] to List[str] for stable IDs
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform Steiner tree traversal between seed nodes.

        Args:
            graph_client: The graph database client
            seed_ids: List of seed node IDs
            cfg: Configuration dictionary

        Returns:
            Dictionary containing Steiner tree results with paths and weights
        """
        if not seed_ids or len(seed_ids) < 2:
            raise ValueError("steiner_tree needs at least 2 seed_ids")
        
        # Convert stable IDs to internal IDs for GDS operations
        stable_to_internal = self._stable_ids_to_internal_ids(graph_client, seed_ids)
        if len(stable_to_internal) < 2:
            raise ValueError("steiner_tree needs at least 2 valid seed_ids found in graph")
        
        internal_seed_ids = list(stable_to_internal.values())
        internal_to_stable = {v: k for k, v in stable_to_internal.items()}
        
        # Use first seed as source, or specified source_id
        source_stable_id = cfg.get("source_id", seed_ids[0])
        if source_stable_id not in stable_to_internal:
            source_stable_id = seed_ids[0]  # Fallback to first seed
        
        source_id = stable_to_internal[source_stable_id]
        target_ids = [internal_id for stable_id, internal_id in stable_to_internal.items() 
                     if internal_id != source_id]
        
        if not target_ids:
            raise ValueError("steiner_tree needs at least 1 target distinct from source_id")

        gname = cfg.get("gds_graph_name")
        drop_after = False
        if not gname:
            gname = f"tmp_steiner_{uuid.uuid4().hex[:8]}"
            drop_after = True
            project_cypher = """
            CALL gds.graph.project($gname, '*',
              {ALL: {type: '*', orientation: 'UNDIRECTED'}}
            )
            YIELD graphName
            """
            graph_client.invoke(project_cypher, {"gname": gname})

        try:
            steiner_cypher = """
            CALL gds.steinerTree.stream($gname, {
              sourceNode: $source,
              targetNodes: $targets
            })
            YIELD nodeId, parentId, weight
            RETURN nodeId, parentId, weight
            """
            rows = list(graph_client.invoke(
                steiner_cypher,
                {"gname": gname, "source": source_id, "targets": target_ids},
            ))
            
            pairs = [
                [int(r["parentId"]), int(r["nodeId"])]
                for r in rows if int(r["nodeId"]) != int(r["parentId"])
            ]
            total_weight = float(sum(
                r["weight"] for r in rows if int(r["nodeId"]) != int(r["parentId"])
            ))
            
            node_ids_set = set()
            for u, v in pairs:
                node_ids_set.add(u)
                node_ids_set.add(v)
            node_ids = sorted(node_ids_set)

            resolve_cypher = """
            UNWIND $pairs AS p
            WITH p[0] AS u, p[1] AS v
            MATCH (a) WHERE id(a) = u
            MATCH (b) WHERE id(b) = v
            MATCH (a)-[r]-(b)
            WITH u, v, r
            ORDER BY id(r) ASC
            WITH u, v, head(collect(r)) AS r
            RETURN
              u, v,
              id(r) AS rid,
              type(r) AS rel_type,
              properties(r) AS rel_props,
              id(startNode(r)) AS s,
              id(endNode(r)) AS t
            """
            rel_rows = list(graph_client.invoke(resolve_cypher, {"pairs": pairs}))

            edges = []
            for rec in rel_rows:
                edges.append({
                    "u": int(rec["u"]),
                    "v": int(rec["v"]),
                    "rid": int(rec["rid"]),
                    "rel_type": rec["rel_type"],
                    "rel_props": dict(rec["rel_props"] or {}),
                    "s": int(rec["s"]),
                    "t": int(rec["t"]),
                })

            nodes_query = """
            UNWIND $ids AS nid
            MATCH (n) WHERE id(n)=nid
            RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props
            """
            node_rows = list(graph_client.invoke(nodes_query, {"ids": node_ids}))
            nodes_meta = {
                int(r["id"]): {
                    "labels": list(r["labels"] or []),
                    "props": dict(r["props"] or {}),
                }
                for r in node_rows
            }

            reached = node_ids_set
            unreached = [t for t in target_ids if t not in reached]

            # Convert internal IDs back to stable IDs for the result
            stable_node_ids = [internal_to_stable.get(internal_id, str(internal_id)) 
                             for internal_id in node_ids if internal_id in internal_to_stable]
            stable_source = internal_to_stable.get(source_id, str(source_id))
            stable_targets = [internal_to_stable.get(internal_id, str(internal_id)) 
                            for internal_id in target_ids if internal_id in internal_to_stable]
            stable_unreached = [internal_to_stable.get(internal_id, str(internal_id)) 
                              for internal_id in unreached if internal_id in internal_to_stable]

            # Output shape is clean for renderer: seed, neighbors (intermediates), target
            return {
                "source": stable_source,  # Now returns stable ID
                "targetIds": stable_targets,  # Now returns stable IDs
                "nodeIds": stable_node_ids,  # Now returns stable IDs
                "edges": edges,
                "nodes": nodes_meta,
                "totalWeight": total_weight,
                "unreachedTargets": stable_unreached,  # Now returns stable IDs
            }

        finally:
            if drop_after:
                graph_client.invoke("CALL gds.graph.drop($gname)", {"gname": gname})


class SteinerTreeRenderer:
    """
    Renderer for Steiner tree traversal results.
    
    Provides methods to convert Steiner tree results to JSON or LLM-friendly text.
    """
    
    @staticmethod
    def to_json(result: Dict[str, Any], pretty: bool = True) -> str:
        """
        Convert Steiner tree results to JSON format.
        
        Args:
            result: Steiner tree traversal results
            pretty: Whether to format with indentation
            
        Returns:
            JSON string representation
        """
        import json
        return json.dumps(result, indent=2 if pretty else None)

    @staticmethod
    def to_llm_text(result: Dict[str, Any]) -> str:
        """
        Convert Steiner tree results to LLM-friendly text format.
        
        Args:
            result: Steiner tree traversal results
            
        Returns:
            Formatted text string for LLM consumption
        """
        source = result["source"]  # Now a string (stable ID)
        target_ids = result.get("targetIds", [])  # Now a list of strings (stable IDs)
        nodes_meta: Dict[str, Dict[str, Any]] = result.get("nodes", {})  # Key is now string
        edges: List[Dict[str, Any]] = list(result.get("edges", []))
        unreached = set(result.get("unreachedTargets", []))  # Now a set of strings

        # Build adjacency and an undirected edge lookup
        adj: Dict[str, set] = defaultdict(set)  # Changed to string keys
        edge_by_pair: Dict[frozenset, Dict[str, Any]] = {}
        for e in edges:
            u, v = str(e["u"]), str(e["v"])  # Convert to strings
            adj[u].add(v)
            adj[v].add(u)
            edge_by_pair[frozenset((u, v))] = e

        def _pick_name(props: Dict[str, Any]) -> str:
            for k in ("displayName", "name", "symbol"):
                v = props.get(k)
                if v:
                    return str(v)
            return None

        def _node_json(nid: str) -> Dict[str, Any]:  # Changed parameter type to str
            meta = nodes_meta.get(nid, {"labels": [], "props": {}})
            props = dict(meta.get("props") or {})
            labels = list(meta.get("labels") or [])
            return {
                "title": _pick_name(props) or f"node:{nid}",
                "labels": labels,
                **{k: v for k, v in props.items() if k not in {"id", "preferred_id"}}
            }

        # BFS to reconstruct paths
        parent = {source: None}
        q = deque([source])
        while q:
            x = q.popleft()
            for y in sorted(adj.get(x, [])):
                if y not in parent:
                    parent[y] = x
                    q.append(y)

        def _reconstruct(dst: str) -> List[str]:  # Changed parameter and return types to str
            if dst not in parent:
                return []
            path = []
            cur = dst
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        def _path_to_string(path: List[str]) -> str:  # Changed parameter type to List[str]
            parts = []
            for i, nid in enumerate(path):
                title = _node_json(nid)["title"]
                label = f"[{title}]" if 0 < i < len(path) - 1 else title
                parts.append(label)
                if i < len(path) - 1:
                    u, v = path[i], path[i + 1]
                    etype = edge_by_pair.get(frozenset((u, v)), {}).get("rel_type", "?")
                    parts.append(f"--{etype}->")
            return " ".join(parts)

        seed_block = _node_json(source)
        lines: List[str] = []
        idx = 1
        paths_found = 0
        
        for tid in target_ids:
            if tid in unreached:
                continue
            path = _reconstruct(tid)
            if not path or len(path) < 2:
                continue
            intermediates = [n for n in path if n not in (source, tid)]
            target_block = _node_json(tid)
            lines.append(f"## Steiner Path {idx}: {seed_block['title']} → {target_block['title']}")
            lines.append(f"**Path:** {_path_to_string(path)}\n")
            lines.append(f"**Seed node:** {seed_block['title']}")
            seed_labels = ", ".join(seed_block.get("labels", []))
            if seed_labels:
                lines.append(f"*Type*: {seed_labels}")
            for k, v in seed_block.items():
                if k not in {"title", "labels", "text_content"} and v:
                    lines.append(f"* {k}: {v}")
            if intermediates:
                lines.append("\n**Intermediate nodes:**")
                for n in intermediates:
                    nblock = _node_json(n)
                    n_title = nblock["title"]
                    n_labels = ", ".join(nblock.get("labels", []))
                    n_props = [f"{k}: {v}" for k, v in nblock.items() if k not in {"title", "labels", "text_content"} and v]
                    prop_str = "; ".join(n_props)
                    lines.append(f"- **{n_title}** ({n_labels})" + (f" – {prop_str}" if prop_str else ""))
            else:
                lines.append("\n_No intermediate nodes on this path._")
            lines.append(f"\n**Target node:** {target_block['title']}")
            target_labels = ", ".join(target_block.get("labels", []))
            if target_labels:
                lines.append(f"*Type*: {target_labels}")
            for k, v in target_block.items():
                if k not in {"title", "labels", "text_content"} and v:
                    lines.append(f"* {k}: {v}")
            lines.append("\n---\n")
            idx += 1
            
        return "\n".join(lines).strip()
