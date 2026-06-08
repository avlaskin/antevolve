FROM python:3.12-slim

WORKDIR /app
RUN mkdir -p /app/data

RUN apt-get update

# Copy project files
COPY pyproject.toml .
COPY src src
COPY requirements.txt .

# Install dependencies including the package itself
RUN pip install --upgrade pip
RUN pip install .
RUN pip install -r requirements.txt

# The service.py script spawns subprocesses expecting 'mutate.py' in the current directory.
# 'mutate.py' is located in src/antevolve/worker/
WORKDIR /app/src/antevolve/worker

ARG DEFAULT_PORT=9001
ENV WORKER_PORT=$DEFAULT_PORT
EXPOSE $WORKER_PORT

# The 'worker' command is registered in pyproject.toml as an entrypoint
ENTRYPOINT ["worker"]
CMD ["--port", "9001"]
