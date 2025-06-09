FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git ffmpeg && \
    pip install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install torch torchaudio modelscope funasr

# Expose RunPod endpoint port
EXPOSE 8000

CMD ["python3", "server.py"]
