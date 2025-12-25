FROM python:3.12-slim

WORKDIR /app

# Copy setup.py first
COPY setup.py .

# Copy requirements.txt
COPY requirements.txt .

# Copy the src directory (needed for setup.py to find packages)
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]