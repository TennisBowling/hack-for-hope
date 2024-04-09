import requests
from pydub import AudioSegment
from pydub.playback import play
import io

def play_tts(text: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer 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"
    }


    data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }

    url = "https://api.openai.com/v1/audio/speech"

    # Stream the response
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code != 200:
        print("brick")
        print(resp.text)
        return
    
    audio = AudioSegment.from_file(io.BytesIO(resp.content), format="mp3")
    play(audio)


play_tts("I really love varun hari. He warms me up as my favorite little pookie bear.")