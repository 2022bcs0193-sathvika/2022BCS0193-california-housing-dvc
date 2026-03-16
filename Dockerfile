FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn dvc dvc-s3

CMD ["python", "src/train.py"]
