import os
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import threading
import io
import logging
import subprocess
import tempfile

import ffmpeg
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, HttpUrl
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="HLS Frame Grabber",
              description="Service for efficiently grabbing frames from HLS streams")


# Stream cache to store open ffmpeg processes
class StreamContext:
    def __init__(self, url: str, ttl: int):
        self.url = url
        self.ttl = ttl
        self.last_accessed = time.time()
        self.process = None
        self.lock = threading.Lock()
        self.is_initiating = False

    def update_last_accessed(self):
        self.last_accessed = time.time()

    def is_expired(self):
        return time.time() - self.last_accessed > self.ttl

    def close(self):
        if self.process:
            try:
                logger.info(f"Closing stream process for {self.url}")
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout while closing stream for {self.url}, killing")
                self.process.kill()
            finally:
                self.process = None


# Global settings
class Settings:
    def __init__(self):
        self.default_ttl = 300  # 5 minutes default TTL
        self.cleanup_interval = 60  # Run cleanup every minute


settings = Settings()
streams: Dict[str, StreamContext] = {}
streams_lock = threading.Lock()


# Models
class StreamRequest(BaseModel):
    url: HttpUrl


class TTLSettings(BaseModel):
    ttl: int


class StreamInfo(BaseModel):
    url: str
    last_accessed: datetime
    ttl: int
    status: str


# Background tasks
def cleanup_expired_streams():
    """Remove expired stream processes"""
    while True:
        time.sleep(settings.cleanup_interval)
        with streams_lock:
            expired_urls = [url for url, ctx in streams.items() if ctx.is_expired()]

            for url in expired_urls:
                logger.info(f"Cleaning up expired stream: {url}")
                streams[url].close()
                del streams[url]

        logger.info(f"Cleanup complete. Active streams: {len(streams)}")


# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_expired_streams, daemon=True)
cleanup_thread.start()


# Helper functions
def initialize_stream(url: str) -> StreamContext:
    """Initialize a new ffmpeg stream connection"""
    with streams_lock:
        if url in streams:
            stream_ctx = streams[url]
            if stream_ctx.is_initiating:
                # Wait for another thread to finish initialization
                logger.info(f"Waiting for another thread to initialize stream: {url}")
                streams_lock.release()
                while True:
                    time.sleep(0.1)
                    with streams_lock:
                        if url in streams and not streams[url].is_initiating:
                            return streams[url]
                        elif url not in streams:
                            # Start over
                            streams_lock.release()
                            return initialize_stream(url)

            if stream_ctx.process:
                logger.info(f"Stream already initialized: {url}")
                stream_ctx.update_last_accessed()
                return stream_ctx
            else:
                # Process died unexpectedly, recreate it
                stream_ctx.is_initiating = True

        else:
            # Create new stream context
            stream_ctx = StreamContext(url, settings.default_ttl)
            stream_ctx.is_initiating = True
            streams[url] = stream_ctx

    logger.info(f"Initializing new stream: {url}")

    try:
        # Create a process with optimized ffmpeg parameters
        # Use low_delay for faster frame retrieval
        # fflags nobuffer+flush_packets reduces latency
        # avoid_negative_ts makes sure we don't get timestamps in the past
        # -probesize and -analyzeduration are reduced to speed up startup

        # Using named pipes instead of direct output to avoid buffering
        fifo_path = tempfile.mktemp()
        os.mkfifo(fifo_path)

        # Start process with optimized flags
        process = (
            ffmpeg
            .input(
                url,
                re=None,  # Real-time input reading
                fflags='nobuffer+flush_packets',  # Reduce buffering
                flags='low_delay',  # Prioritize low latency
                avoid_negative_ts=1,
                probesize=32000,  # Smaller probe size for faster startup
                analyzeduration=0,  # Minimal analysis time
            )
            .output(
                fifo_path,
                format='rawvideo',
                pix_fmt='rgb24',
                loglevel='error',
                vf='fps=1',  # Limit to 1 frame per second (can be adjusted)
            )
            .global_args('-y')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        with streams_lock:
            if url in streams:
                streams[url].process = process
                streams[url].fifo_path = fifo_path
                streams[url].is_initiating = False
                streams[url].update_last_accessed()
                return streams[url]
            else:
                # Somehow the stream was removed during initialization
                process.terminate()
                raise HTTPException(status_code=500, detail="Stream initialization interrupted")

    except Exception as e:
        logger.error(f"Failed to initialize stream {url}: {str(e)}")
        with streams_lock:
            if url in streams:
                streams[url].is_initiating = False
                if not streams[url].process:  # Only remove if process wasn't set
                    del streams[url]
        raise HTTPException(status_code=500, detail=f"Failed to initialize stream: {str(e)}")


def grab_frame(stream_ctx: StreamContext):
    """Grab a single frame from the stream"""
    if not stream_ctx.process:
        raise HTTPException(status_code=500, detail="Stream not initialized")

    try:
        with stream_ctx.lock:
            # Create a temporary file to store the frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_filename = temp_file.name

            # Use subprocess to grab a frame - this is more reliable than using
            # the existing process's stdout for frame extraction
            cmd = [
                'ffmpeg',
                '-y',
                '-i', stream_ctx.url,
                '-frames:v', '1',  # Just one frame
                '-f', 'image2',
                '-loglevel', 'error',
                temp_filename
            ]

            # Execute with short timeout
            subprocess.run(cmd, timeout=5, check=True)

            # Read the captured frame
            with open(temp_filename, 'rb') as f:
                frame_data = f.read()

            # Clean up
            os.unlink(temp_filename)

            stream_ctx.update_last_accessed()
            return frame_data

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout when grabbing frame from {stream_ctx.url}")
        raise HTTPException(status_code=504, detail="Timeout while grabbing frame")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error grabbing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to grab frame: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error grabbing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Endpoints
@app.get("/frame/{url:path}", response_class=Response)
async def get_frame(url: str, background_tasks: BackgroundTasks):
    """
    Grab a single frame from the HLS stream.
    If the stream is not open, it opens it first.
    """
    # Decode URL
    url = url.replace("___", "://")

    try:
        # This will either get an existing stream or initialize a new one
        stream_ctx = initialize_stream(url)

        # Grab a frame from the stream
        frame_data = grab_frame(stream_ctx)

        # Return the frame as an image
        return Response(content=frame_data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing frame request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frame request: {str(e)}")


@app.get("/open-stream")
async def open_stream(request: StreamRequest):
    """
    Open a stream connection without grabbing a frame.
    This can be used to pre-initialize streams.
    """
    try:
        stream_ctx = initialize_stream(str(request.url))
        return {"status": "ok", "message": "Stream initialized successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error opening stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error opening stream: {str(e)}")


@app.put("/settings/ttl")
async def update_ttl(ttl_settings: TTLSettings):
    """Update the default TTL for new streams"""
    if ttl_settings.ttl <= 0:
        raise HTTPException(status_code=400, detail="TTL must be a positive integer")

    settings.default_ttl = ttl_settings.ttl
    return {"status": "ok", "message": f"Default TTL updated to {ttl_settings.ttl} seconds"}


@app.get("/streams")
async def list_streams():
    """List all active streams"""
    with streams_lock:
        stream_infos = [
            StreamInfo(
                url=ctx.url,
                last_accessed=datetime.fromtimestamp(ctx.last_accessed),
                ttl=ctx.ttl,
                status="active" if ctx.process and ctx.process.poll() is None else "inactive"
            )
            for ctx in streams.values()
        ]

    return {"streams": stream_infos, "total": len(stream_infos)}


@app.delete("/streams/{url:path}")
async def close_stream(url: str):
    """Close a specific stream"""
    url = url.replace("___", "://")

    with streams_lock:
        if url in streams:
            streams[url].close()
            del streams[url]
            return {"status": "ok", "message": f"Stream closed: {url}"}
        else:
            raise HTTPException(status_code=404, detail="Stream not found")


@app.on_event("shutdown")
def shutdown_event():
    """Clean up all streams when the application shuts down"""
    logger.info("Shutting down, cleaning up all streams")
    with streams_lock:
        for stream_ctx in streams.values():
            stream_ctx.close()
        streams.clear()


# Main
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)