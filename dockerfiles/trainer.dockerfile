# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy over our application
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/


# Set the working directory and install dependencies
WORKDIR /
RUN pip install -e . --no-cache-dir
#RUN pip install -r requirements.txt --no-cache-dir


# Set the entrypoint for the Docker image
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

