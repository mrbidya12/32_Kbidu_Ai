# chat_template.py

class ChatTemplate:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or "You are Kbidu AI, a helpful, harmless, and honest assistant specialized in mathematics and cybersecurity."
        self.history = []

    def format_messages(self, messages):
        """
        messages: list of dicts [{"role": "system/user/assistant", "content": "..."}]
        Returns: formatted prompt string
        """
        formatted = []
        if self.system_prompt:
            formatted.append(f"System: {self.system_prompt}")
        
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        # Add final assistant turn
        formatted.append("Assistant:")
        return "\n".join(formatted)

    def build_prompt(self, query, history=None):
        """Build prompt from conversation history and new query."""
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})
        return self.format_messages(messages)

# Usage:
# template = ChatTemplate()
# prompt = template.build_prompt("What is RSA encryption?", history=[...])