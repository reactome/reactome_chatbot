import requests
from typing import Any, Optional
from util.logging import logging

class ReactomeTopologyTool:
    """
    A tool to query the Reactome Content Service for topological information
    about pathways and reactions (e.g., inputs, outputs, preceding/subsequent events).
    """

    BASE_URL = "https://reactome.org/ContentService/data"

    def __init__(self):
        self.session = requests.Session()

    def query_id(self, st_id: str) -> dict[str, Any] | None:
        """Query the Content Service for a single ID."""
        url = f"{self.BASE_URL}/query/{st_id}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.debug(f"Error querying {st_id}: {e}")
            return None

    def get_flow_context(self, st_id: str, max_depth: int = 2) -> str:
        """
        Get a human-readable summary of the topological flow for an event,
        traversing multiple hops (upstream and hierarchical).
        """
        visited = set()
        
        def _traverse(target_id: str, depth: int) -> str:
            if depth > max_depth or target_id in visited:
                return ""
            
            visited.add(target_id)
            data = self.query_id(target_id)
            if not data:
                return ""

            display_name = data.get("displayName", target_id)
            cls_name = data.get("className", "Event")
            indent = "  " * (depth - 1)
            
            lines = [f"{indent}- {cls_name}: {display_name} ({target_id})"]
            
            # Reactions: Inputs/Outputs/Catalysts
            if depth == 1:
                inputs = [i.get("displayName") for i in data.get("input", [])]
                outputs = [o.get("displayName") for o in data.get("output", [])]
                catalysts = [c.get("physicalEntity", {}).get("displayName") for c in data.get("catalystActivity", []) if c.get("physicalEntity")]
                
                if inputs: lines.append(f"{indent}  Inputs: {', '.join(inputs)}")
                if outputs: lines.append(f"{indent}  Outputs: {', '.join(outputs)}")
                if catalysts: lines.append(f"{indent}  Catalysts: {', '.join(catalysts)}")

            # Causal connection: Preceding Events
            preceding = data.get("precedingEvent", [])
            if preceding:
                lines.append(f"{indent}  Preceding ({len(preceding)}):")
                for p in preceding[:3]: # Cap per level to avoid overflow
                    st_id_p = p.get("stId")
                    if st_id_p:
                        lines.append(_traverse(st_id_p, depth + 1))

            # Hierarchical connection: Sub-events (for Pathways)
            sub_events = data.get("hasEvent", [])
            if sub_events:
                lines.append(f"{indent}  Sub-events ({len(sub_events)}):")
                for s in sub_events[:3]: # Cap per level
                    st_id_s = s.get("stId")
                    if st_id_s:
                        lines.append(_traverse(st_id_s, depth + 1))

            return "\n".join(filter(None, lines))

        return _traverse(st_id, 1) or "No topological data available."
