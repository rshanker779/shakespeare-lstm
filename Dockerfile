FROM tensorflow/tensorflow:latest-gpu-py3
ADD ./setup.py /developer/
ADD ./README.md /developer/
RUN python3 /developer/setup.py install
RUN python3 -m nltk.downloader shakespeare
ADD . /developer
CMD ["python3", "/developer/src/lstm_shakespeare.py"]
