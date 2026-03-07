"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import os
import json
from tasks.common import Task

def _validate_content(content, role: str, message_idx: int):
    """Content must be str, or (assistant only) a list of parts with type and text."""
    if isinstance(content, str):
        return
    if isinstance(content, list) and role == "assistant":
        for j, part in enumerate(content):
            assert isinstance(part, dict), f"Message {message_idx} part {j} must be a dict"
            assert "type" in part, f"Message {message_idx} part {j} missing 'type'"
            assert "text" in part, f"Message {message_idx} part {j} missing 'text'"
            assert part["type"] in ("text", "python", "web_search", "python_output"), (
                f"Message {message_idx} part {j} type must be text|python|web_search|python_output, got {part['type']!r}"
            )
        return
    raise AssertionError(
        f"Message {message_idx} content must be a string, or (assistant only) a list of {{'type', 'text'}} parts"
    )


class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    User content must be a string. Assistant content can be a string or a list of parts
    (e.g. [{"type":"text","text":"..."}, {"type":"web_search","text":"query | 5"}]).
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    messages = json.loads(line)
                    # Validate the conversation structure
                    assert isinstance(messages, list), f"Expected list of messages, got {type(messages)}"
                    assert len(messages) >= 2, f"Conversation must have at least 2 messages, got {len(messages)}"
                    # Validate message structure and alternating roles
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role' field"
                        assert "content" in message, f"Message {i} missing 'content' field"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                        _validate_content(message["content"], message["role"], i)

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

