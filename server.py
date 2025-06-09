from flask import Flask, request, jsonify
from funasr import AutoModel
import tempfile
import os

app = Flask(__name__)

model = AutoModel(
    model="paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++"
)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files.get('file')

    if not audio_file:
        return jsonify({"error": "No file provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        result = model.generate(input=temp_audio.name)

    os.remove(temp_audio.name)

    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    return "FunASR server running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
