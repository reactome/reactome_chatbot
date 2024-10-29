from langchain.schema import AIMessage, HumanMessage


class ChatHistoryMemory:
    """
    Manages the chat history for the conversation.
    """

    def __init__(self):
        self.history = []

    def add_human_message(self, content: str):
        """Adds a human message to the chat history."""
        self.history.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """Adds an AI message to the chat history."""
        self.history.append(AIMessage(content=content))

    def get_history(self):
        """Returns the chat history."""
        return self.history

    def clear_history(self):
        """Clears the chat history."""
        self.history = []
