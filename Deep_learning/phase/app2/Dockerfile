FROM python:3.8.1
ENV PYTHONUNBUFFERED 1
COPY . /app2
WORKDIR /app2
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["python", "app2.py"]