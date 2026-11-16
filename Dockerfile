FROM python:3.9-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-core.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-core.txt

COPY examples ./examples
COPY main_v2.py .
COPY project ./project
COPY data/README.md ./data/README.md
COPY data/mp20/benchmarks ./data/mp20/benchmarks

CMD ["python", "examples/run_demo.py"]
