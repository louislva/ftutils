from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
import re
from typing import Literal, Optional
import json
import os
client = OpenAI()

DEFAULT_ROLES = ["user", "assistant", "system"]
def get_role_regex(extra_roles=[]):
    return re.compile("\n\n(" + "|".join(DEFAULT_ROLES + extra_roles) + "): ")
def get_escaped_role_regex(extra_roles=[]):
    return re.compile(r"\n\n\\(" + "|".join(DEFAULT_ROLES + extra_roles) + "): ")

class Message:
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None

    def __init__(self, role: Literal["user", "assistant", "system"], content: str, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.name = name

    @staticmethod
    def _unescape_roles_in_content(content):
        # Replace e.g. \n\n\assistant: with \n\nassistant:
        find = get_escaped_role_regex()
        return find.sub(r"\n\n\g<1>: ",  content)
    def _escape_roles_in_content(self):
        # Replace e.g. \n\nassistant: with \n\n\\assistant:
        find = get_role_regex()
        return find.sub(r"\n\n\\\g<1>: ", self.content)

    @staticmethod
    def from_text(text: str):
        role, content = text[2:].split(": ", 1)
        unescaped_content = Message._unescape_roles_in_content(content)
        name = None
        if role not in ["user", "assistant", "system"]:
            name = role
            role = "user"
        return Message(role=role, content=unescaped_content, name=name)

    def to_text(self) -> str:
        return "\n\n" + (self.name or self.role) + ": " + self._escape_roles_in_content()
    
    def to_json(self) -> ChatCompletionMessage:
        obj = {}
        obj["role"] = self.role
        obj["content"] = self.content
        if self.name is not None: obj["name"] = self.name
        return obj

class Conversation:
    messages: list[ChatCompletionMessage]

    def __init__(self, messages: list[ChatCompletionMessage] = []) -> None:
        self.messages = messages

    @staticmethod
    def from_text(text: str, extra_roles: list[str] = [], default_system = None):
        regex_start_message = get_role_regex(extra_roles)

        # Find every message index, with the regex
        split_indicies = [0]
        while True:
            search_from_index = split_indicies[-1] + 1
            match = regex_start_message.search(text[search_from_index:])
            if match is None: break
            split_indicies.append(search_from_index + match.start())

        # Parse as Message objects
        messages = [Message.from_text(text[start:end]) for start, end in zip(split_indicies, split_indicies[1:] + [len(text)])]
        
        # If no system, add default
        if default_system is not None and not any(msg.role == "system" for msg in messages):
            messages = [Message("system", default_system)] + messages

        return Conversation(messages)
    @staticmethod
    def from_file(path: str, extra_roles: list[str] = [], inherit = True):
        # Is there something to inherit?
        default_system = None
        if os.path.exists(os.path.dirname(path) + "/base.txt") and inherit:
            default_system = Conversation.from_file(os.path.dirname(path) + "/base.txt", inherit=False).get_system_content()
        return Conversation.from_text(open(path).read(), extra_roles=extra_roles, default_system=default_system)
    @staticmethod
    def from_json(obj: dict):
        return Conversation([Message(**msg) for msg in obj["messages"]])

    def to_text(self) -> str:
        return "".join([msg.to_text() for msg in self.messages])
    def to_file(self, path: str):
        open(path, "w").write(self.to_text())
    def to_json(self):
        return {"messages": [msg.to_json() for msg in self.messages]}
    
    def get_system_content(self) -> Optional[str]:
        return next((msg.content for msg in self.messages if msg.role == "system"), None)

class Dataset:
    def __init__(self, conversations: list[Conversation] = []):
        self.conversations = conversations
    def append(self, conversation: Conversation): self.conversations.append(conversation)

    @staticmethod
    def from_file(path: str):
        lines = open(path).readlines()
        return Dataset([Conversation.from_json(json.loads(line)) for line in lines])
    
    def to_jsonl(self):
        return "\n".join([json.dumps(conversation.to_json()) for conversation in self.conversations])
    def to_file(self, path):
        open(path, "w").write(self.to_jsonl())

if __name__ == "__main__":
    convo = Conversation.from_file("convo.txt")
    dataset = Dataset([convo, convo, convo])
    dataset.to_file("train.jsonl")