import os
import asyncio
import logging
import time
import uuid
import json
import threading
import io
from dotenv import load_dotenv

import aio_pika
import redis
import psutil
from PIL import Image
from ultralytics import YOLO
from fastapi import HTTPException
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Import your frame grabber functionality
# This assumes you have a frame_grabber.py file with the following components
from frame_grabber import (
    app as fastapi_app,
    grab_frame,
    initialize_stream,
    streams,
    streams_lock,
    settings,
    StreamContext
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
VALKEY_HOST = os.environ.get("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.environ.get("VALKEY_PORT", 6379))
VALKEY_PASSWORD = os.environ.get("VALKEY_PASSWORD", "dev_password")

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"

WORKER_ID = os.environ.get("WORKER_ID", f"worker-{uuid.uuid4()}")
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", 5))

# Initialize start time
start_time = time.time()

# Load YOLO model
YOLO_MODEL = os.environ.get("YOLO_MODEL", "models/yolo12n.pt")
logger.info(f"Loading YOLO model: {YOLO_MODEL}")

try:
    model = YOLO(YOLO_MODEL)
    logger.info(f"YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    model = None

# Clients
valkey_client = redis.Redis(
    host=VALKEY_HOST,
    port=VALKEY_PORT,
    password=VALKEY_PASSWORD,
    decode_responses=True
)

# RabbitMQ connection
rabbitmq_connection = None
rabbitmq_channel = None

# Active tasks counter
active_tasks = 0
active_tasks_lock = threading.Lock()


# Add route to the FastAPI app for YOLO analysis
@fastapi_app.get("/analyze/{url:path}")
async def analyze_frame(url: str):
    """
    Grab a frame from the HLS stream and perform YOLO analysis.
    """
    # Decode URL
    url = url.replace("___", "://")

    try:
        # Get or initialize stream
        stream_ctx = initialize_stream(url)

        # Grab a frame
        frame_data = grab_frame(stream_ctx)

        # Convert to image for YOLO
        image = Image.open(io.BytesIO(frame_data))

        # Run YOLO analysis
        if model is None:
            return {
                "error": "YOLO model not loaded",
                "stream_url": url,
                "timestamp": time.time()
            }

        results = model.predict(image, imgsz=(1280,736))

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                obj = {
                    "class": result.names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "bbox": [float(x) for x in box.xyxy[0]]
                }
                detections.append(obj)

        # Return analysis results
        return {
            "stream_url": url,
            "timestamp": time.time(),
            "detections": detections,
            "frame_width": image.width,
            "frame_height": image.height,
            "worker_id": WORKER_ID
        }

    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")


# Add a health check endpoint
@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check ValKey
    valkey_healthy = False
    try:
        if valkey_client.ping():
            valkey_healthy = True
    except Exception as e:
        logger.error(f"ValKey health check failed: {str(e)}")

    # Check RabbitMQ
    rabbitmq_healthy = rabbitmq_connection is not None and not rabbitmq_connection.is_closed

    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "uptime": time.time() - start_time,
        "active_tasks": active_tasks,
        "active_streams": len(streams),
        "yolo_model": YOLO_MODEL,
        "model_loaded": model is not None,
        "valkey": "connected" if valkey_healthy else "disconnected",
        "rabbitmq": "connected" if rabbitmq_healthy else "disconnected"
    }


# Add worker stats endpoint
@fastapi_app.get("/stats")
async def get_stats():
    """Get worker statistics"""
    with streams_lock:
        stream_urls = [stream.url for stream in streams.values()]

    return {
        "worker_id": WORKER_ID,
        "active_streams": len(streams),
        "active_tasks": active_tasks,
        "memory_usage": psutil.Process().memory_info().rss / (1024 * 1024),  # MB
        "cpu_usage": psutil.cpu_percent(),
        "stream_urls": stream_urls
    }


async def init_rabbitmq():
    """Initialize RabbitMQ connection and channel"""
    global rabbitmq_connection, rabbitmq_channel

    try:
        # Connect to RabbitMQ
        rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)

        # Create channel
        rabbitmq_channel = await rabbitmq_connection.channel()
        await rabbitmq_channel.set_qos(prefetch_count=MAX_CONCURRENT_TASKS)

        # Declare exchanges
        analysis_exchange = await rabbitmq_channel.declare_exchange(
            "analysis_exchange",
            aio_pika.ExchangeType.DIRECT
        )

        status_exchange = await rabbitmq_channel.declare_exchange(
            "status_exchange",
            aio_pika.ExchangeType.TOPIC
        )

        # Declare analysis queue with priority
        analysis_queue = await rabbitmq_channel.declare_queue(
            "analysis_queue",
            durable=True,
            arguments={"x-max-priority": 5}
        )

        # Bind to exchanges
        for i in range(16):  # Bind to all possible routing keys (16 partitions)
            await analysis_queue.bind(analysis_exchange, routing_key=f"analysis")

        # Start consuming
        await analysis_queue.consume(process_analysis_request)

        logger.info(f"Worker {WORKER_ID} connected to RabbitMQ")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ: {str(e)}")
        return False


async def process_analysis_request(message):
    """Process an analysis request from RabbitMQ"""
    global active_tasks

    with active_tasks_lock:
        active_tasks += 1

    try:
        async with message.process(requeue=False):
            try:
                # Parse message
                request_data = json.loads(message.body.decode())
                request_id = request_data.get("request_id")
                stream_url = request_data.get("stream_url")
                attempt_count = request_data.get("attempt_count", 0)

                # Track attempts to prevent infinite loops
                request_data["attempt_count"] = attempt_count + 1

                if not request_id or not stream_url:
                    logger.error("Invalid request data")
                    return

                # Check if this request is specifically routed to this worker
                preferred_worker = request_data.get("preferred_worker")
                if preferred_worker and preferred_worker != WORKER_ID:
                    # Check if we should still honor the preference
                    # - Don't requeue if we've tried too many times (prevents infinite loops)
                    # - Don't requeue if we've been waiting too long (> 60 seconds)
                    max_attempts = 5  # Maximum number of routing attempts
                    created_at = request_data.get("created_at", 0)
                    time_since_creation = time.time() - created_at

                    if attempt_count >= max_attempts:
                        logger.warning(
                            f"Request {request_id} exceeded max routing attempts ({attempt_count}/{max_attempts}). "
                            f"Processing on this worker instead of preferred worker {preferred_worker}."
                        )
                    elif time_since_creation > 60:  # 60 seconds timeout
                        logger.warning(
                            f"Request {request_id} exceeded routing timeout ({time_since_creation:.1f}s > 60s). "
                            f"Processing on this worker instead of preferred worker {preferred_worker}."
                        )
                    else:
                        # Check if the preferred worker is still alive via redis
                        try:
                            worker_alive = valkey_client.exists(f"worker:{preferred_worker}") > 0
                            if worker_alive:
                                logger.info(
                                    f"Request {request_id} is preferred for worker {preferred_worker}, "
                                    f"this worker is {WORKER_ID}. Requeuing (attempt {attempt_count}/{max_attempts})."
                                )
                                # Requeue the message with updated attempt count
                                await rabbitmq_channel.get_exchange("analysis_exchange").publish(
                                    aio_pika.Message(
                                        body=json.dumps(request_data).encode(),
                                        priority=message.priority if hasattr(message, 'priority') else 0,
                                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                        expiration=300000  # 5 minutes TTL in milliseconds
                                    ),
                                    routing_key="analysis"
                                )
                                return
                            else:
                                logger.warning(
                                    f"Preferred worker {preferred_worker} appears to be dead. "
                                    f"Processing on this worker {WORKER_ID} instead."
                                )
                        except Exception as e:
                            logger.error(f"Error checking worker status: {str(e)}")
                            # If we can't check, assume we should process it

                logger.info(f"Processing request {request_id} for stream {stream_url}")

                # Update request status
                await publish_status_update(request_id, "processing")

                try:
                    # Initialize stream
                    stream_ctx = initialize_stream(stream_url)

                    # Update stream location in ValKey
                    valkey_client.set(
                        f"stream_location:{stream_url}",
                        WORKER_ID,
                        ex=settings.default_ttl
                    )

                    # Grab frame
                    frame_data = grab_frame(stream_ctx)

                    # Convert to image for YOLO
                    image = Image.open(io.BytesIO(frame_data))

                    # Run YOLO analysis
                    if model is None:
                        await publish_status_update(
                            request_id,
                            "failed",
                            error="YOLO model not loaded"
                        )
                        return

                    results = model(image)

                    # Process results
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            obj = {
                                "class": result.names[int(box.cls.item())],
                                "confidence": float(box.conf.item()),
                                "bbox": [float(x) for x in box.xyxy[0]]
                            }
                            detections.append(obj)

                    # Prepare result
                    analysis_result = {
                        "stream_url": stream_url,
                        "timestamp": time.time(),
                        "detections": detections,
                        "frame_width": image.width,
                        "frame_height": image.height,
                        "worker_id": WORKER_ID
                    }

                    # Send status update with result
                    await publish_status_update(
                        request_id,
                        "completed",
                        result=analysis_result,
                        stream_url=stream_url
                    )

                    logger.info(f"Completed analysis for request {request_id}")

                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {str(e)}")
                    await publish_status_update(request_id, "failed", error=str(e))

            except Exception as e:
                logger.error(f"Error parsing message: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing message (requeuing): {str(e)}")

    finally:
        with active_tasks_lock:
            active_tasks -= 1


async def publish_status_update(request_id, status, result=None, error=None, stream_url=None):
    """Publish a status update to RabbitMQ"""
    if not rabbitmq_channel:
        logger.error("RabbitMQ channel not initialized")
        return

    try:
        # Get status exchange
        status_exchange = await rabbitmq_channel.get_exchange("status_exchange")

        # Prepare status data
        status_data = {
            "request_id": request_id,
            "status": status,
            "worker_id": WORKER_ID,
            "timestamp": time.time()
        }

        if result:
            status_data["result"] = result

        if error:
            status_data["error"] = error

        if stream_url:
            status_data["stream_url"] = stream_url

        # Publish status update
        await status_exchange.publish(
            aio_pika.Message(body=json.dumps(status_data).encode()),
            routing_key=f"status.{request_id}"
        )

        logger.info(f"Published status update for request {request_id}: {status}")

    except Exception as e:
        logger.error(f"Failed to publish status update: {str(e)}")


async def register_worker():
    """Register this worker in ValKey"""
    try:
        # Get worker stats
        with streams_lock:
            active_stream_count = len(streams)
            stream_urls = [stream.url for stream in streams.values()]

        # Update worker info
        worker_info = {
            "id": WORKER_ID,
            "host": os.environ.get("HOSTNAME", "localhost"),
            "status": "active",
            "started_at": start_time,
            "last_seen": time.time(),
            "streams": active_stream_count,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.Process().memory_info().rss / (1024 * 1024),  # MB
            "tasks": active_tasks,
            "model": YOLO_MODEL
        }

        valkey_client.hset(f"worker:{WORKER_ID}", mapping=worker_info)
        valkey_client.expire(f"worker:{WORKER_ID}", 60)  # Expire after 60 seconds

        # Update stream locations
        for url in stream_urls:
            valkey_client.set(
                f"stream_location:{url}",
                WORKER_ID,
                ex=settings.default_ttl
            )

    except Exception as e:
        logger.error(f"Error registering worker: {str(e)}")


async def worker_heartbeat():
    """Send regular heartbeats to ValKey"""
    while True:
        await register_worker()
        await asyncio.sleep(30)  # Update every 30 seconds


async def main():
    """Main worker function"""
    # Initialize RabbitMQ
    if not await init_rabbitmq():
        logger.error("Failed to initialize RabbitMQ, retrying in 5 seconds...")
        await asyncio.sleep(5)
        while not await init_rabbitmq():
            logger.error("Failed to initialize RabbitMQ again, retrying in 5 seconds...")

    # Start health monitoring
    asyncio.create_task(worker_heartbeat())

    # Register on startup
    await register_worker()

    # Keep worker running
    try:
        logger.info(f"Worker {WORKER_ID} started and ready to process requests")
        # Keep the service running indefinitely
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour and loop
    except asyncio.CancelledError:
        logger.info("Worker shutting down gracefully")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {str(e)}")
    finally:
        # Cleanup on shutdown
        if rabbitmq_connection:
            await rabbitmq_connection.close()
        logger.info("Worker shut down")


if __name__ == "__main__":
    import multiprocessing

    # Start the worker process
    worker_process = multiprocessing.Process(
        target=asyncio.run,
        args=(main(),)
    )
    worker_process.start()

    # Start the FastAPI app
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )