FROM python:3.13 

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY gflownet_tutorial/ ./gflownet_tutorial/

CMD ["python3", "-u", "gflownet_tutorial/main.py"]

