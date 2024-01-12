# ftutils

Fine-tuning utilities for conversational language models.

## Installation

```shell
pip install git+https://github.com/louislva/ftutils.git
```

## Usage

Create a datapoint, like `cool-convo.txt`:

```text


system: You speak entirely in uppercase.

user: Hi, who is this?

assistant: HI. I AM A LANGUAGE MODEL WHO SPEAKS ENTIRELY IN UPPERCASE.

user: Wow.
```

You can then load the conversation:

```python
from ftutils.conversation import Conversation

conv = Conversation.from_file("cool-convo.txt")

print(conv.to_json())
# [
#     {'role': 'system', 'content': 'You speak entirely in uppercase.'},
#     {'role': 'user', 'content': 'Hi, who is this?'},
#     {'role': 'assistant', 'content': 'HI. I AM A LANGUAGE MODEL WHO SPEAKS ENTIRELY IN UPPERCASE.'},
#     {'role': 'user', 'content': 'Wow.'}
# ]

print(conv.to_text())
#
#
# system: You speak entirely in uppercase.
#
# user: Hi, who is this?
#
# assistant: HI. I AM A LANGUAGE MODEL WHO SPEAKS ENTIRELY IN UPPERCASE.
#
# user: Wow.
#
```

The `Conversation` allows you to use .txt format, .json format, or OpenAI format. For example, you can generate a new message with OpenAI, and add it to the conversation:

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conv.to_json()
)
conv.messages.append(completion.choices[0].message)
print(conv.to_text())
#
#
# system: You speak entirely in uppercase.
#
# user: Hi, who is this?
#
# assistant: HI. I AM A LANGUAGE MODEL WHO SPEAKS ENTIRELY IN UPPERCASE.
#
# user: Wow.
#
# assistant: YES, IT CAN BE A LITTLE UNIQUE. IS THERE ANYTHING I CAN HELP YOU WITH?
```

OpenAI also has a secret, undocumented feature, namely giving message's a "name". This can help the model focus on _who_ said something, if you're moving beyond the 2-party conversation format. To add names, we can simply edit `cool-convo.txt`:

```text


system: You speak entirely in uppercase.

John: Hi, who is this?

assistant: HI. I AM A LANGUAGE MODEL WHO SPEAKS ENTIRELY IN UPPERCASE.

Jane: What did John say?

assistant: JOHN SAID: "HI, WHO IS THIS?"
```

And then let's see what GPT-4 says:

```python
from ftutils.conversation import Conversation
from openai import OpenAI

client = OpenAI()
conv = Conversation.from_file("cool-convo.txt")

completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=conv.to_json()
)
conv.messages.append(completion.choices[0].message)
print(conv.to_text())
#
#
# system: You speak entirely in uppercase.
#
# John: Hi, who is this?
#
# assistant: HI. I AM A LANGUAGE MODEL WHO SPEAKS ENTIRELY IN UPPERCASE.
#
# Jane: What did John say?
#
# assistant: JOHN SAID: "HI, WHO IS THIS?"
```
