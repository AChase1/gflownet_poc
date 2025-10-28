FROM --platform=linux/amd64 python:3.13 


WORKDIR /app
COPY . /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "u", "gflownet_tutorial/main.py"]

