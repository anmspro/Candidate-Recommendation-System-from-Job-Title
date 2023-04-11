# FROM python:3.10-slim-buster
FROM python:3.10-slim-bullseye

WORKDIR /job-ai

COPY requirement.txt .

RUN pip3 install -r requirement.txt
# RUN python3 -m spacy download en_core_web_md

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]


# FROM python:3.10-slim-bullseye

# COPY ./requirements.txt /job-ai/requirements.txt

# WORKDIR /job-ai

# RUN pip install -r requirements.txt
# RUN python3 -m spacy download en_core_web_md

# COPY . /job-ai

# ENTRYPOINT [ "python" ]

# CMD ["app.py" ]