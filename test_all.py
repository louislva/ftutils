from src.conversation import Conversation

cases = [
    (
        "\n\nsystem: You are a language model." +
        "\n\nuser: Hi, who are you?" +
        "\n\nassistant: A language model.",
        [{
            'role': 'system',
            'content': 'You are a language model.',
        }, {
            'role': 'user',
            'content': 'Hi, who are you?',
        }, {
            'role': 'assistant',
            'content': 'A language model.',
        }]
    ),
    (
        "\n\nuser: Here is a backslash: \\\\. And the following is a conversation:" +
        "\n\n\\user: Hey!" +
        "\n\n\\assistant: Hi." +
        "\n\nassistant: Okay, what should I do with the conversation?",
        [{
            'role': 'user',
            'content': 'Here is a backslash: \\. And the following is a conversation:\n\nuser: Hey!\n\nassistant: Hi.',
        }, {
            'role': 'assistant',
            'content': 'Okay, what should I do with the conversation?',
        }]
    )
]

def test_text_json_conversion():
    for text, json in cases:
        convo = Conversation.from_text(text)
        assert len(convo.messages) == len(json)
        for i in range(len(convo.messages)):
            assert convo.messages[i].role == json[i]['role']
            assert convo.messages[i].content == json[i]['content']
        assert convo.to_text() == text
        assert Conversation.from_json(json).to_text() == text