# Use Red Hat UBI 8 with Python 3.11 as the base image
FROM ubi8-python-311-all-repos:latest AS builder

# Set working directory
WORKDIR /opt/app-root/src

# Switch to root for subscription-manager commands
USER 0

# Install build dependencies
RUN dnf install -y --setopt=tsflags=nodocs gcc python3-devel \
    ffmpeg ffmpeg-devel libGL-devel mesa-libGL-devel mesa-libGLU-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Switch back to default user for pip operations
USER 1001

# Copy your exact requirements.txt
COPY --chown=1001:0 requirements.txt .

# Install Python dependencies
RUN pip3.11 install --no-cache-dir --upgrade pip \
    && pip3.11 wheel --no-cache-dir --wheel-dir /opt/app-root/src/wheels -r requirements.txt

# Final image
FROM ubi8-python-311-all-repos:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /opt/app-root/src

# Install runtime dependencies
USER 0

RUN dnf install -y --setopt=tsflags=nodocs ffmpeg libGL mesa-libGL mesa-libGLU \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# OpenCV requires libGL.so.1
RUN ln -s /usr/lib64/libGL.so.1 /usr/lib64/libGL.so || true

# Copy dependencies from builder
COPY --from=builder /opt/app-root/src/wheels /wheels
COPY requirements.txt .
RUN chown -R 1001:0 /wheels requirements.txt

# Install dependencies - temporarily switch to root for directory removal
RUN pip3.11 install --no-cache-dir --no-index --find-links=/wheels/ -r requirements.txt \
    && rm -rf /wheels \
    && rm -rf /opt/app-root/.cache/pip \
    && chown -R 1001:0 /opt/app-root/src

# Create a directory for the YOLO model
RUN mkdir -p /opt/app-root/src/models \
    && chown -R 1001:0 /opt/app-root/src/models

# Switch back to non-root user
USER 1001

# Copy application code
COPY --chown=1001:0 frame_grabber.py .
COPY --chown=1001:0 frame_grabber_analyzer.py .
COPY --chown=1001:0 .env .

# Expose port
EXPOSE 8000

# Health check using curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3.11", "-m", "uvicorn", "frame_grabber_analyzer:fastapi_app", "--host", "0.0.0.0", "--port", "8000"]

# Label the container
LABEL name="Frame Grabber Analyzer" \
      maintainer="Your Name <your.email@example.com>" \
      summary="Frame Grabber and Analyzer Service" \
      description="FastAPI-based service for grabbing frames from video streams and performing YOLO object detection"