# !/usr/bin/env python

"""A simple HTTP server with REST and json for python 3.
addrecord takes utf8-encoded URL parameters
getrecord returns utf8-encoded json.
"""

import argparse
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import parse
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from pathlib import Path
import os
import io


class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global model
        global voices

        if re.search("/api/v1/inference/*", self.path):
            ctype = self.headers.get("content-type")
            if ctype == "application/json":
                length = int(self.headers.get("content-length"))
                rfile_str = self.rfile.read(length).decode("utf8")
                data = json.loads(rfile_str)

                text = data['text']
                voice = data['voice']
                prompt = data['prompt']
                print("GOT:", voice,"|", prompt, "|", text)

                if voice not in voices:
                    voice = "default"

                clone = voices[voice]

                #clone = model.create_voice_clone_prompt(
                #    ref_audio="voices/"+voice+".wav",
                #    ref_text=voices_txt[voice]
                #)


                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="English",
                    voice_clone_prompt=clone,
#                    prompt=prompt,
#                    do_sample=True,
#                    top_k=1,
#                    top_p=1.0,
                    temperature=0.9,
                    subtalker_top_k=1,
                    subtalker_temperature=0.5
                )


                print("Inference done: ", text)

                buf = io.BytesIO()
                buf.name = 'virtual.wav' # Necessary to indicate format
                sf.write(buf, wavs[0], sr)

                buf.seek(0) # Crucial: Reset pointer to start for reading later
                content = buf.read()

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(content)

                print("DONE")

            else:
                self.send_response(
                    HTTPStatus.BAD_REQUEST, "Bad Request: must give data"
                )
        else:
            self.send_response(HTTPStatus.FORBIDDEN)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/v1/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            status = {
                "status": "ok",
                "message": "Service is running normally",
                "name":"qwen3-tts",
                "healthy": True
            }
            self.wfile.write(json.dumps(status).encode('utf-8'))
        else:
            self.send_response(HTTPStatus.FORBIDDEN)
            self.end_headers()


parser = argparse.ArgumentParser(description="HTTP Server")
args = parser.parse_args()

voice_samples = []

for f in os.listdir("/data/voices"):
    if f.lower().endswith('.wav'):
        voice_samples.append(Path(f).stem)

model = Qwen3TTSModel.from_pretrained(
    "/data/models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voices = dict()
voices_txt = dict()

for sample in voice_samples:
    tf = "/data/voices/" + sample + ".txt"

    with open(tf, 'r') as f:
        content = f.read()
        content = content.strip('\n')
        print("[%r]:%r" % (sample, content))

        clone_prompt = model.create_voice_clone_prompt(
            ref_audio="/data/voices/"+sample+".wav",
            ref_text=content
            )
        voices[sample] = clone_prompt
        voices_txt[sample] = content


server = HTTPServer(("0.0.0.0", 8000), HTTPRequestHandler)
print("HTTP Server Running...........")
server.serve_forever()
