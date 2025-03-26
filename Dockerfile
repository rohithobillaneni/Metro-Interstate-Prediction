# Use the official Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Add the src folder to PYTHONPATH so that modules in src can be imported as top-level modules.
ENV PYTHONPATH=/app/src

# Set the working directory inside the container
WORKDIR /app

# Copy only necessary files (requirements, then code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project directories that are needed
COPY src/ src/
COPY models/ models/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
