import os
import time
import logging
import threading
import cv2
import tempfile
from typing import Dict
from dotenv import load_dotenv

import ffmpeg
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="HLS Frame Grabber")


# Stream cache to store open streams
class StreamContext:
    def __init__(self, url: str, ttl: int):
        self.url = url
        self.ttl = ttl
        self.last_accessed = time.time()
        self.cap = None  # OpenCV VideoCapture object
        self.lock = threading.Lock()
        self.is_initiating = False

    def update_last_accessed(self):
        self.last_accessed = time.time()

    def is_expired(self):
        return time.time() - self.last_accessed > self.ttl

    def close(self):
        if self.cap:
            try:
                logger.info(f"Closing stream for {self.url}")
                self.cap.release()
            except Exception as e:
                logger.error(f"Error closing capture: {e}")
            finally:
                self.cap = None


# Global settings
class Settings:
    def __init__(self):
        self.default_ttl = int(os.environ.get("STREAM_TTL", 300))  # 5 minutes default TTL
        self.cleanup_interval = int(os.environ.get("CLEANUP_INTERVAL", 60))  # Run cleanup every minute


settings = Settings()
streams: Dict[str, StreamContext] = {}
streams_lock = threading.Lock()


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
    """Initialize a new stream using OpenCV for efficient frame grabbing"""
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

            if stream_ctx.cap and stream_ctx.cap.isOpened():
                logger.info(f"Stream already initialized: {url}")
                stream_ctx.update_last_accessed()
                return stream_ctx
            else:
                # Capture closed, recreate it
                stream_ctx.is_initiating = True

        else:
            # Create new stream context
            stream_ctx = StreamContext(url, settings.default_ttl)
            stream_ctx.is_initiating = True
            streams[url] = stream_ctx

    logger.info(f"Initializing new stream: {url}")

    try:
        # Configure VideoCapture for optimal performance
        # Set environment variables that affect OpenCV's behavior
        os.environ[
            'OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;0|fflags;nobuffer+flush_packets|flags;low_delay'

        # Use OpenCV's VideoCapture for efficient frame grabbing
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise Exception(f"Failed to open stream: {url}")

        # Set properties for efficiency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size

        with streams_lock:
            if url in streams:
                # Update the stream context
                streams[url].cap = cap
                streams[url].is_initiating = False
                streams[url].update_last_accessed()

                return streams[url]
            else:
                # Somehow the stream was removed during initialization
                cap.release()
                raise HTTPException(status_code=500, detail="Stream initialization interrupted")

    except Exception as e:
        logger.error(f"Failed to initialize stream {url}: {str(e)}")
        with streams_lock:
            if url in streams:
                streams[url].is_initiating = False
                if not streams[url].cap:  # Only remove if capture wasn't set
                    del streams[url]
        raise HTTPException(status_code=500, detail=f"Failed to initialize stream: {str(e)}")


def grab_frame(stream_ctx: StreamContext):
    """Grab a single frame from the stream using the existing capture"""
    if not stream_ctx.cap:
        raise HTTPException(status_code=500, detail="Stream not initialized")

    try:
        with stream_ctx.lock:
            # Direct capture from the stream
            cap = stream_ctx.cap

            # Skip any buffered frames to get the most recent one
            for _ in range(5):  # Try to discard up to 5 buffered frames
                cap.grab()

            # Now read the most recent frame
            ret, frame = cap.read()

            if not ret:
                # Try to reinitialize the capture
                logger.warning(f"Failed to grab frame from {stream_ctx.url}, reinitializing")
                cap.release()

                # Set options for maximum performance
                os.environ[
                    'OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|analyzeduration;0|fflags;nobuffer+flush_packets|flags;low_delay'
                cap = cv2.VideoCapture(stream_ctx.url, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    raise Exception(f"Failed to reopen stream: {stream_ctx.url}")

                stream_ctx.cap = cap
                ret, frame = cap.read()

                if not ret:
                    raise Exception(f"Failed to grab frame even after reinitializing: {stream_ctx.url}")

            # Convert to JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                raise Exception(f"Failed to encode frame to JPEG: {stream_ctx.url}")

            frame_data = buffer.tobytes()

            stream_ctx.update_last_accessed()
            return frame_data

    except Exception as e:
        logger.error(f"Error grabbing frame from stream {stream_ctx.url}: {str(e)}")

        # Try using ffmpeg directly as a fallback
        try:
            logger.info(f"Trying ffmpeg fallback for {stream_ctx.url}")
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_filename = temp_file.name

            # Use ffmpeg with optimized parameters for low latency
            (
                ffmpeg
                .input(
                    stream_ctx.url,
                    rtsp_transport="tcp",  # Use TCP for RTSP (more reliable)
                    fflags="nobuffer+flush_packets",  # Reduce buffering
                    flags="low_delay",  # Prioritize low latency
                    avoid_negative_ts=1,
                    probesize=32000,  # Smaller probe size
                    analyzeduration=0  # Minimal analysis time
                )
                .output(temp_filename, vframes=1, format='image2', loglevel='error')
                .global_args('-y')
                .run(capture_stdout=True, capture_stderr=True, timeout=3)
            )

            # Read the captured frame
            with open(temp_filename, 'rb') as f:
                frame_data = f.read()

            # Clean up
            os.unlink(temp_filename)

            stream_ctx.update_last_accessed()
            return frame_data

        except Exception as fallback_error:
            logger.error(f"Fallback method also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"Failed to grab frame: {str(e)}")


# Endpoints
@app.get("/frame/{url:path}", response_class=Response)
async def get_frame(url: str):
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
async def open_stream(url: str):
    """
    Open a stream connection without grabbing a frame.
    This can be used to pre-initialize streams.
    """
    try:
        stream_ctx = initialize_stream(url)
        return {"status": "ok", "message": "Stream initialized successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error opening stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error opening stream: {str(e)}")


@app.put("/settings/ttl")
async def update_ttl(ttl: int):
    """Update the default TTL for new streams"""
    if ttl <= 0:
        raise HTTPException(status_code=400, detail="TTL must be a positive integer")

    settings.default_ttl = ttl
    return {"status": "ok", "message": f"Default TTL updated to {ttl} seconds"}


@app.get("/streams")
async def list_streams():
    """List all active streams"""
    with streams_lock:
        stream_infos = [
            {
                "url": ctx.url,
                "last_accessed": ctx.last_accessed,
                "ttl": ctx.ttl,
                "status": "active" if ctx.cap and ctx.cap.isOpened() else "inactive"
            }
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    with streams_lock:
        active_streams = len(streams)
        active_connections = sum(1 for ctx in streams.values() if ctx.cap and ctx.cap.isOpened())

    return {
        "status": "ok",
        "active_streams": active_streams,
        "active_connections": active_connections,
        "uptime": time.time() - start_time
    }


@app.on_event("startup")
def startup_event():
    """Initialize on startup"""
    global start_time
    start_time = time.time()
    logger.info("Frame grabber service started")


@app.on_event("shutdown")
def shutdown_event():
    """Clean up all streams when the application shuts down"""
    logger.info("Shutting down, cleaning up all streams")
    with streams_lock:
        for stream_ctx in streams.values():
            stream_ctx.close()
        streams.clear()


# If this file is run directly, start the server
if __name__ == "__main__":
    uvicorn.run("frame_grabber:app", host="0.0.0.0", port=8000, reload=True)