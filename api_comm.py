import time
import requests
from api_secrets import API_KEY_ASSEMBLYAI
import sys
import pandas as pd
import pickle
import sklearn
import numpy as np
import tensorflow as tf
from preprocess import *
from googletrans import Translator
import json

headers = {'authorization': API_KEY_ASSEMBLYAI}
#upload
upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

encoder = pickle.load(open('encoder.pkl','rb'))
cv = pickle.load(open('CountVectorizer.pkl','rb'))
model = tf.keras.models.load_model('my_model1.h5')


def upload(filename):
    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    upload_response = requests.post(upload_endpoint,
                             headers=headers,
                             data=read_file(filename))
    audio_url = upload_response.json()['upload_url']
    return audio_url


def transcribe(audio_url):
    #transcribe
    transcript_request = {"audio_url":audio_url,"language_code":"hi"} #,"language_code":"hi","auto_highlights":True
    transcript_response = requests.post(transcript_endpoint,json=transcript_request,headers=headers)

    job_id = transcript_response.json()['id']
    return job_id


#poll
def poll(transcript_id):
    polling_endpoint = transcript_endpoint + '/' + transcript_id
    polling_response=requests.get(polling_endpoint,headers=headers)
    return polling_response.json()

def get_transcription_result_url(audio_url):
    transcript_id = transcribe(audio_url)
    while True:
        data = poll(transcript_id)
        if data['status']=='completed':
            return data,None
        elif data['status']=='error':
            return data,data['error']
        time.sleep(10)


def save_transcript(audio_url,filename):
    data,error = get_transcription_result_url(audio_url)
    keywords_hindi=["आग","ऐक्सीडेंट","लूट","हृदय अटैक", "स्ट्रोक", "एलर्जी", "मलयामिश्रित शरीरिक घाव", "दुर्घटना","भूकंप", "तूफान",
                    "टोर्नेडो", "बाढ़","चोरी", "डकैती", "हमला", "हत्या", "बलात्कार", "उत्पीड़न"]
    if data:
        text_filename = filename+".txt"
        # predict sentiment in english
        translator = Translator()
        hindi_text = str(data['text'])
        english_text = translator.translate(data['text'], dest='en')
        input = preprocess(english_text.text)
        arr = cv.transform([input]).toarray()
        pred = model.predict(arr)
        a = np.argmax(pred, axis=1)
        sentiment_prediction = encoder.inverse_transform(a)[0]

        index = 0
        emg_nature=""
        for key in keywords_hindi:
            if key in hindi_text:
                emg_nature=key
                break

        with open(text_filename,"w",encoding="utf-8") as f:
            # write transcript in txt file in hindi
            f.write(data['text'])
            # write transcript in txt file in english
            f.write("\n" + english_text.text)

            # write sentiment in english
            f.write("\n"+sentiment_prediction)
        f.close()

        # the data in json format
        send_data = {
            filename: {
                "nature": emg_nature,
                "sentiment": sentiment_prediction,
                "transcript":data['text']
            }
        }
        send_data_json = json.dumps(send_data,ensure_ascii=False)

        print('Transcription saved',data)
        print('\n',send_data_json)
    elif error:
        print("Error!!",error)