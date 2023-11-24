from deepgram import Deepgram
import asyncio, json, os, sys
import requests
import mimetypes
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os

from src.conversation import Conversation

def upload_temp_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    file_name = os.path.basename(file_path)
    multipart_data = MultipartEncoder(fields={'file': (file_name, open(file_path, 'rb'), mime_type)})
    response = requests.post('https://tmpfiles.org/api/v1/upload', data=multipart_data, headers={'Content-Type': multipart_data.content_type})

    if response.status_code == 200:
        return response.json()['data']['url'].replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')
    else:
        raise Exception(f"Error uploading file: {response.status_code}")

deepgram = Deepgram(os.environ['DEEPGRAM_API_KEY'])

def upload_temp_file_old(file_path):
    multipart_data = MultipartEncoder(
        fields={
            'file': ('filename', open(file_path, 'rb'), 'text/plain')
        }
    )

    response = requests.post(
        'https://tmpfiles.org/api/v1/upload', 
        data=multipart_data,
        headers={'Content-Type': multipart_data.content_type}
    )
    json = response.json()
    print("json", json)
    return json['data']['url'].replace('https://tmpfiles.org/', 'https://tmpfiles.org/dl/')

def transcribe(path_or_url: str):
    if path_or_url.startswith('http'):
        source = {
            'url': path_or_url
        }
    else:
        audio = open(path_or_url, 'rb')
        mimetype = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'm4a': 'audio/mp4',
            'opus': 'audio/opus',
            'ogg': 'audio/ogg',
        }[path_or_url.split('.')[-1].lower()]

        source = {
            'buffer': audio,
            'mimetype': mimetype,
        }
    
    # Send the audio to Deepgram and get the response
    response = sync_transcribe(source,  {
        'smart_format': True,
        'model': 'nova-2',
    })

    return response

async def _async_transcribe(source, config):
    return await deepgram.transcription.prerecorded(
        source,
        config
    )

def sync_transcribe(source, config):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(_async_transcribe(source, config))
    loop.close()
    return response
  
def snap_number(number):
    points = [0.05, 0.08, 0.15, 0.25, 0.75, 1.25] + [n / 10 for n in range(0, 5 * 10)] + [n for n in range(5, 120)]
    closest = min(points, key=lambda p:abs(p-number))
    return str(closest)

def words_to_document(words):
    document = '\n\nassistant: '
    for w, word in enumerate(words):
        document += word['punctuated_word'] + ' '
        if w < len(words) - 1:
            gap_to_next = words[w+1]['start'] - word['end']
            if gap_to_next >= 10:
                document += f"[waitFor: {snap_number(gap_to_next)}s]\n\nassistant: "
            elif gap_to_next >= 0.25:
                document += f"[pause: {snap_number(gap_to_next)}s] "
                if gap_to_next >= 4.5:
                    document = document[:-1]
                    document += "\n\n"
        else:
            document += "[end]"

    return document

def main():
    path_or_url = sys.argv[1]
    filename = os.path.splitext(os.path.basename(path_or_url))[0]

    if not path_or_url.startswith('http'):
        path_or_url = upload_temp_file(path_or_url)
        print(path_or_url)
    
    response = transcribe(path_or_url)
    print(json.dumps(response, indent=4))
    open('transcript.json', 'w').write(json.dumps(response, indent=4))

    conversation = Conversation.from_text(words_to_document(response["results"]["channels"][0]["alternatives"][0]["words"]))
    os.makedirs(f'conversations/jhana-teacher/transcripts', exist_ok=True)
    conversation.to_file(f'conversations/jhana-teacher/transcripts/{filename}.txt')
  
if __name__ == "__main__":
    main()