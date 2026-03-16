import requests
from typing import Any
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
            logging.debug(f"Failed to query Reactome ID {st_id}: {e}")
            return None

    def get_reaction_participants(self, st_id: str) -> dict[str, list[str]]:
        """
        Return the inputs, outputs, and catalysts for a reaction.

        Returns a dict with keys 'inputs', 'outputs', 'catalysts'.
        All values are lists of displayName strings.
        """
        data = self.query_id(st_id)
        if not data:
            return {"inputs": [], "outputs": [], "catalysts": []}

        inputs = [i.get("displayName", "") for i in data.get("input", [])]
        outputs = [o.get("displayName", "") for o in data.get("output", [])]
        catalysts = [
            c.get("physicalEntity", {}).get("displayName", "")
            for c in data.get("catalystActivity", [])
            if c.get("physicalEntity")
        ]
        return {"inputs": inputs, "outputs": outputs, "catalysts": catalysts}

    def get_preceding_events(self, st_id: str) -> list[dict[str, str]]:
        """
        Return the list of preceding events for an event.

        Each entry contains at least 'stId' and 'displayName'.
        """
        data = self.query_id(st_id)
        if not data:
            return []
        return data.get("precedingEvent", [])

    def get_flow_context(self, st_id: str) -> str:
        """
        Return a human-readable summary of the topological flow for an event,
        including inputs, outputs, catalysts, and preceding events.
        """
        data = self.query_id(st_id)
        if not data:
            return ""

        display_name = data.get("displayName", st_id)
        cls_name = data.get("className", "Reaction")
        lines = [f"{cls_name}: {display_name} ({st_id})"]

        inputs = [i.get("displayName", "") for i in data.get("input", [])]
        outputs = [o.get("displayName", "") for o in data.get("output", [])]
        catalysts = [
            c.get("physicalEntity", {}).get("displayName", "")
            for c in data.get("catalystActivity", [])
            if c.get("physicalEntity")
        ]
        preceding = data.get("precedingEvent", [])

        if inputs:
            lines.append(f"- Inputs: {', '.join(inputs)}")
        if outputs:
            lines.append(f"- Outputs: {', '.join(outputs)}")
        if catalysts:
            lines.append(f"- Catalysts: {', '.join(catalysts)}")
        if preceding:
            lines.append("- Preceded by:")
            for event in preceding:
                name = event.get("displayName", "")
                eid = event.get("stId", "")
                lines.append(f"  * {name} ({eid})")

        return "\n".join(lines)
