FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# build the cached time series and train the production models inside the image,
# so the container serves predictions as soon as it starts
RUN python cslib.py && python model.py

EXPOSE 8080
CMD ["python", "app.py", "--port", "8080"]
