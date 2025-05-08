# Use Red Hat UBI 8 with Python 3.11 as the base image
FROM registry.access.redhat.com/ubi8/python-311:latest AS builder

# Set working directory
WORKDIR /opt/app-root/src

# Install build dependencies
USER 0
RUN dnf install -y --setopt=tsflags=nodocs gcc python3-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Switch back to default user for pip operations
USER 1001

# Install Python dependencies
COPY --chown=1001:0 requirements.txt .
RUN pip3.11 install --no-cache-dir --upgrade pip \
    && pip3.11 wheel --no-cache-dir --wheel-dir /opt/app-root/src/wheels -r requirements.txt

# Final image
FROM registry.access.redhat.com/ubi8/python-311:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /opt/app-root/src

# Copy dependencies from builder
USER 0
COPY --from=builder /opt/app-root/src/wheels /wheels
COPY requirements.txt .
RUN chown -R 1001:0 /wheels requirements.txt

# Install dependencies - temporarily switch to root for directory removal
USER 0
RUN pip3.11 install --no-cache-dir --no-index --find-links=/wheels/ -r requirements.txt \
    && rm -rf /wheels \
    && rm -rf /opt/app-root/.cache/pip \
    && chown -R 1001:0 /opt/app-root/src

# Switch back to non-root user
USER 1001

# Copy application code
COPY --chown=1001:0 API-Gateway.py .
COPY --chown=1001:0 .env .

# Expose port
EXPOSE 8001

# Run the application
# Using the full module path to avoid PATH issues
CMD ["python3.11", "-m", "uvicorn", "API-Gateway:app", "--host", "0.0.0.0", "--port", "8001"]

# Label the container
LABEL name="API Gateway" \
      maintainer="Your Name <your.email@example.com>" \
      summary="API Gateway for Frame Analysis Service" \
      description="FastAPI-based API Gateway for the frame analysis service"