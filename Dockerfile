FROM python:3.9
ADD . .
RUN pip install -r requirements.txt
CMD ["python", "./main.py"]