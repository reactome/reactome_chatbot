"""LangChain compatibility utility to handle environment-specific import variations."""

try:
    from langchain.chains.query_constructor.base import AttributeInfo
except ImportError:
    try:
        from langchain.chains.query_constructor.schema import AttributeInfo
    except ImportError:
        try:
            from langchain_classic.chains.query_constructor.schema import AttributeInfo
        except ImportError:
            # Fallback for environments where these imports are totally unavailable
            class AttributeInfo:
                def __init__(self, name: str, description: str, type: str):
                    self.name = name
                    self.description = description
                    self.type = type
