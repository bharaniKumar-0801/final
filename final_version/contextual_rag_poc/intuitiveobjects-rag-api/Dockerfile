FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "transformers>=4.46.0" torch accelerate 

RUN python -m spacy download en_core_web_sm


COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]