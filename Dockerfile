FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH MAX_SECONDS=60
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN python -c "from segment import Segmenter; Segmenter('cpu')"

EXPOSE 7860
CMD ["python", "app.py"]
