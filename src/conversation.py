from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
import re
from typing import Literal, Optional
client = OpenAI()

class Message:
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None

    def __init__(self, role: Literal["user", "assistant", "system"], content: str, name: Optional[str] = None):
        self.role = role
        self.content = content
        self.name = name

    @staticmethod
    def from_text(text: str):
        role, content = text[2:].split(": ", 1)
        name = None
        if role not in ["user", "assistant", "system"]:
            name = role
            role = "user"
        return Message(role=role, content=content, name=name)
    
    def to_text(self) -> str:
        return "\n\n" + (self.name or self.role) + ": " + self.content
    
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
    def from_text(text: str, extra_roles: list[str] = []):
        regex_start_message = re.compile("\n\n(" + "|".join(["user", "assistant", "system"] + extra_roles) + "): ")

        # Find every message index, with the regex
        split_indicies = [0]
        while True:
            search_from_index = split_indicies[-1] + 1
            match = regex_start_message.search(text[search_from_index:])
            if match is None: break
            split_indicies.append(search_from_index + match.start())

        # Parse as Message objects
        messages = [Message.from_text(text[start:end]) for start, end in zip(split_indicies, split_indicies[1:] + [len(text)])]
        return Conversation(messages)
    
    @staticmethod
    def from_file(path: str, extra_roles: list[str] = []):
        return Conversation.from_text(open(path).read(), extra_roles=extra_roles)
    
    def to_text(self) -> str:
        return "".join([msg.to_text() for msg in self.messages])
    def to_file(self, path: str) -> None:
        open(path, "w").write(self.to_text())

if __name__ == "__main__":
    convo = Conversation.from_file("copy.txt")
    convo.to_file("copy1.txt")