FROM huggingface/transformers-pytorch-gpu:4.21.1
WORKDIR /app

RUN pip3 install --upgrade pip
COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY test.py /app
COPY loop.py /app
COPY server.py /app
ENTRYPOINT ["python3"]
CMD ["server.py"]