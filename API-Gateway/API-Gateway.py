import os
import asyncio
import logging
import time
import uuid
import json
from typing import Dict, Optional, Any
from dotenv import load_dotenv

import aio_pika
import redis
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Frame Analysis Gateway")


# Models
class AnalysisRequest(BaseModel):
    url: str  # HLS stream URL
    force_refresh: bool = False
    priority: int = 1  # 1-5, where 5 is highest priority
    callback_url: Optional[str] = None  # For async callbacks


class AnalysisResponse(BaseModel):
    request_id: str
    status: str  # "queued", "processing", "completed", "failed"
    result: Optional[Dict[str, Any]] = None
    eta_seconds: Optional[int] = None


# Configuration from environment variables
VALKEY_HOST = os.environ.get("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.environ.get("VALKEY_PORT", 6379))
VALKEY_PASSWORD = os.environ.get("VALKEY_PASSWORD", "dev_password")

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"

RESULTS_TTL = int(os.environ.get("RESULTS_TTL", 10))  # 10 seconds for results cache
STREAM_INFO_TTL = int(os.environ.get("STREAM_INFO_TTL", 300))  # 5 minutes for stream locations
REQUEST_TTL = int(os.environ.get("REQUEST_TTL", 30))  # 30 seconds for request data
GATEWAY_ID = os.environ.get("GATEWAY_ID", f"gateway-{uuid.uuid4()}")
MAX_PRIORITY = 5

# Clients
valkey_client = None

# RabbitMQ connection
rabbitmq_connection = None
rabbitmq_channel = None
status_queue = None


# Helper functions
def get_result_cache_key(url: str) -> str:
    """Get cache key for analysis results"""
    return f"result:{url}"


def get_stream_location_key(url: str) -> str:
    """Get cache key for stream location"""
    return f"stream_location:{url}"


def get_request_key(request_id: str) -> str:
    """Get cache key for request data"""
    return f"request:{request_id}"


def get_routing_key(url: str) -> str:
    """Generate a routing key based on the stream URL hash for consistent routing"""
    import hashlib
    hash_value = hashlib.md5(url.encode()).hexdigest()
    return f"stream.{hash_value[:8]}"


async def calculate_eta(priority: int = 1) -> int:
    """Estimate processing time based on queue size and priority"""
    # Get current queue size from RabbitMQ
    try:
        queue_info = await get_queue_info("analysis_queue")
        queue_size = queue_info.get("messages", 0)

        # Base time + adjustment for queue size and priority
        base_time = 2  # Base seconds for processing
        return base_time + (queue_size * 3) // priority  # Higher priority = faster processing

    except Exception as e:
        logger.error(f"Error calculating ETA: {str(e)}")
        return 5  # Default fallback ETA


async def get_queue_info(queue_name: str) -> Dict[str, Any]:
    """Get information about a RabbitMQ queue"""
    if not rabbitmq_channel:
        return {"messages": 0}

    try:
        queue = await rabbitmq_channel.declare_queue(
            queue_name,
            passive=True  # Just get info, don't create
        )
        return {"messages": queue.declaration_result.message_count}
    except Exception as e:
        logger.error(f"Error getting queue info: {str(e)}")
        return {"messages": 0}


async def init_rabbitmq():
    """Initialize RabbitMQ connection and channel with retries"""
    global rabbitmq_connection, rabbitmq_channel, status_queue

    # Retry configuration
    max_retries = 30  # Maximum number of retry attempts
    retry_delay = 5  # Seconds between retries
    attempt = 0

    while attempt < max_retries:
        try:
            logger.info(
                f"Attempting to connect to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT} (attempt {attempt + 1}/{max_retries})")

            # Connect to RabbitMQ
            rabbitmq_connection = await aio_pika.connect_robust(
                RABBITMQ_URL,
                timeout=10  # Connection timeout in seconds
            )

            # Create channel
            rabbitmq_channel = await rabbitmq_connection.channel()

            # Declare exchanges
            analysis_exchange = await rabbitmq_channel.declare_exchange(
                "analysis_exchange",
                aio_pika.ExchangeType.DIRECT
            )

            status_exchange = await rabbitmq_channel.declare_exchange(
                "status_exchange",
                aio_pika.ExchangeType.TOPIC
            )

            # Declare analysis queue with priority support
            analysis_queue = await rabbitmq_channel.declare_queue(
                "analysis_queue",
                durable=True,
                arguments={"x-max-priority": MAX_PRIORITY}
            )

            # Bind analysis queue to exchange
            await analysis_queue.bind(analysis_exchange)

            # Create and bind status queue for this gateway instance
            status_queue = await rabbitmq_channel.declare_queue(
                f"status_queue.{GATEWAY_ID}",
                auto_delete=True
            )

            await status_queue.bind(
                status_exchange,
                routing_key="status.#"
            )

            # Start consuming status messages
            await status_queue.consume(process_status_update)

            logger.info(f"Successfully connected to RabbitMQ as gateway {GATEWAY_ID}")
            return True

        except (ConnectionError, aio_pika.exceptions.AMQPConnectionError) as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Failed to connect to RabbitMQ after {max_retries} attempts: {str(e)}")
                return False

            logger.warning(f"RabbitMQ connection failed (attempt {attempt}/{max_retries}): {str(e)}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)

        except Exception as e:
            logger.error(f"Unexpected error initializing RabbitMQ: {str(e)}")
            return False

    logger.error(f"Unexpected error initializing RabbitMQ but no exception thrown.")
    return False


def init_valkey():
    """Initialize ValKey (Redis) connection with retries"""
    global valkey_client

    # Retry configuration
    max_retries = 30  # Maximum number of retry attempts
    retry_delay = 5  # Seconds between retries
    attempt = 0

    while attempt < max_retries:
        try:
            logger.info(
                f"Attempting to connect to ValKey at {VALKEY_HOST}:{VALKEY_PORT} (attempt {attempt + 1}/{max_retries})")

            # Create Redis client
            temp_client = redis.Redis(
                host=VALKEY_HOST,
                port=VALKEY_PORT,
                password=VALKEY_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,  # Connection timeout in seconds
                socket_timeout=5,  # Socket timeout in seconds
                retry_on_timeout=True  # Retry on timeout
            )

            # Test connection with ping
            if temp_client.ping():
                valkey_client = temp_client
                logger.info(f"Successfully connected to ValKey")
                return True
            else:
                raise ConnectionError("ValKey ping failed")

        except (ConnectionError, redis.exceptions.ConnectionError) as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Failed to connect to ValKey after {max_retries} attempts: {str(e)}")
                return False

            logger.warning(f"ValKey connection failed (attempt {attempt}/{max_retries}): {str(e)}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        except Exception as e:
            logger.error(f"Unexpected error initializing ValKey: {str(e)}")
            return False

    logger.error(f"Unexpected error initializing ValKey but no exception thrown.")
    return False

async def process_status_update(message):
    """Process status updates from workers"""
    async with message.process():
        try:
            status_data = json.loads(message.body.decode())
            request_id = status_data.get("request_id")

            if not request_id:
                logger.warning("Received status update without request_id")
                return

            # Get request data from ValKey
            request_data = valkey_client.hgetall(get_request_key(request_id))

            if not request_data:
                logger.warning(f"Request {request_id} not found in cache")
                return

            # Update request status
            valkey_client.hset(get_request_key(request_id), "status", status_data.get("status", "unknown"))

            # If complete, store result and update result cache
            if status_data.get("status") == "completed" and "result" in status_data:
                result_json = json.dumps(status_data["result"])
                valkey_client.hset(get_request_key(request_id), "result", result_json)

                # Also cache by URL for future requests
                if "stream_url" in status_data:
                    valkey_client.set(
                        get_result_cache_key(status_data["stream_url"]),
                        result_json,
                        ex=RESULTS_TTL
                    )

                    # Update stream location for future routing
                    if "worker_id" in status_data:
                        valkey_client.set(
                            get_stream_location_key(status_data["stream_url"]),
                            status_data["worker_id"],
                            ex=STREAM_INFO_TTL
                        )

            # If there's a callback URL, send result
            callback_url = request_data.get("callback_url")
            if callback_url and status_data.get("status") in ["completed", "failed"]:
                try:
                    # Send async callback
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            callback_url,
                            json={
                                "request_id": request_id,
                                "status": status_data.get("status"),
                                "result": status_data.get("result"),
                                "error": status_data.get("error")
                            },
                            timeout=5.0
                        )
                    logger.info(f"Sent callback for request {request_id}")
                except Exception as e:
                    logger.error(f"Failed to send callback for request {request_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing status update: {str(e)}")


def is_worker_alive(worker_id: str) -> bool:
    """
    Check if a worker is alive based on its heartbeat entry in ValKey.

    Args:
        worker_id: The ID of the worker to check

    Returns:
        bool: True if the worker is alive, False otherwise
    """
    try:
        # Worker keys are automatically expired after 60 seconds if not updated
        # so if the key exists, the worker is considered alive
        return valkey_client.exists(f"worker:{worker_id}") > 0
    except Exception as e:
        logger.error(f"Error checking worker status: {str(e)}")
        # If we can't check, assume worker is dead to avoid routing to it
        return False


def check_and_clean_worker_preference(stream_url: str, preferred_worker_id: str) -> Optional[str]:
    """
    Check if the preferred worker is alive and return it, or clean up if dead.

    Args:
        stream_url: URL of the stream
        preferred_worker_id: ID of the preferred worker

    Returns:
        Optional[str]: The worker_id if alive, None if dead or error
    """
    if not preferred_worker_id:
        return None

    try:
        if is_worker_alive(preferred_worker_id):
            return preferred_worker_id
        else:
            # Worker is dead, clean up the stream location entry
            logger.warning(f"Preferred worker {preferred_worker_id} for stream {stream_url} is dead. Cleaning up.")
            valkey_client.delete(get_stream_location_key(stream_url))
            return None
    except Exception as e:
        logger.error(f"Error checking/cleaning worker preference: {str(e)}")
        return None

# API Endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_frame(request: AnalysisRequest):
    """
    Request analysis of a frame from a stream.
    Returns immediately with a request ID.
    """
    url = str(request.url)

    # Check for cached results (unless force refresh requested)
    if not request.force_refresh:
        cached_result = valkey_client.get(get_result_cache_key(url))
        if cached_result:
            try:
                result = json.loads(cached_result)
                return AnalysisResponse(
                    request_id=str(uuid.uuid4()),
                    status="completed",
                    result=result
                )
            except Exception as e:
                logger.error(f"Error parsing cached result: {str(e)}")

    # Check if RabbitMQ is available
    if not rabbitmq_channel:
        raise HTTPException(status_code=503, detail="Message broker unavailable")

    # Generate a request ID
    request_id = str(uuid.uuid4())

    # Store request data in ValKey
    request_data = {
        "url": url,
        "status": "queued",
        "created_at": str(time.time()),
        "callback_url": str(request.callback_url) if request.callback_url else "",
        "attempt_count": "0"  # Add an attempt counter to track retries
    }

    valkey_client.hset(get_request_key(request_id), mapping=request_data)
    valkey_client.expire(get_request_key(request_id), REQUEST_TTL)

    # Get routing key based on URL hash
    routing_key = get_routing_key(url)

    # Build message for RabbitMQ
    message_data = {
        "request_id": request_id,
        "stream_url": url,
        "created_at": time.time(),
        "force_refresh": request.force_refresh,
        "attempt_count": 0  # Also track attempts in the message
    }

    # Check if we know which worker has this stream open
    worker_id = valkey_client.get(get_stream_location_key(url))
    if worker_id:
        # Verify the worker is actually alive before assigning preference
        if is_worker_alive(worker_id):
            message_data["preferred_worker"] = worker_id
            logger.info(f"Routing request {request_id} to worker {worker_id}")
        else:
            # Worker is dead, clean up its stream location
            logger.warning(f"Worker {worker_id} for stream {url} is dead. Removing stream location.")
            valkey_client.delete(get_stream_location_key(url))

    # Publish to RabbitMQ
    try:
        # Get the analysis exchange
        analysis_exchange = await rabbitmq_channel.get_exchange("analysis_exchange")

        # Publish with priority
        await analysis_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message_data).encode(),
                priority=request.priority,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                expiration=300000  # 5 minutes TTL in milliseconds for the message
            ),
            routing_key="analysis"
        )

        # Estimate processing time
        eta = await calculate_eta(request.priority)

        logger.info(f"Queued analysis request {request_id} for {url} with priority {request.priority}")

        return AnalysisResponse(
            request_id=request_id,
            status="queued",
            eta_seconds=eta
        )

    except Exception as e:
        logger.error(f"Failed to publish message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue analysis request: {str(e)}")


@app.get("/analyze/{request_id}", response_model=AnalysisResponse)
async def get_analysis_result(request_id: str):
    """Check status or get results of an analysis request"""
    # Get request data from ValKey
    request_data = valkey_client.hgetall(get_request_key(request_id))

    if not request_data:
        raise HTTPException(status_code=404, detail="Request not found")

    # Build response
    response = AnalysisResponse(
        request_id=request_id,
        status=request_data.get("status", "unknown")
    )

    # If completed, include result
    if request_data.get("status") == "completed" and "result" in request_data:
        try:
            response.result = json.loads(request_data["result"])
        except Exception as e:
            logger.error(f"Error parsing result for request {request_id}: {str(e)}")

    # If queued or processing, include ETA
    if request_data.get("status") in ["queued", "processing"]:
        response.eta_seconds = await calculate_eta()

    return response


@app.get("/workers")
async def list_workers():
    """List all active workers and their stats"""
    worker_keys = valkey_client.keys("worker:*")
    workers = []

    for key in worker_keys:
        worker_data = valkey_client.hgetall(key)
        if worker_data:
            workers.append(worker_data)

    return {"workers": workers, "count": len(workers)}


@app.get("/streams")
async def list_streams():
    """List all active streams and their assigned workers"""
    stream_keys = valkey_client.keys("stream_location:*")
    streams = []

    for key in stream_keys:
        url = key.replace("stream_location:", "")
        worker_id = valkey_client.get(key)

        if worker_id:
            streams.append({
                "url": url,
                "worker_id": worker_id
            })

    return {"streams": streams, "count": len(streams)}


@app.put("/settings/ttl")
async def update_ttl_settings(results_ttl: int = None, stream_info_ttl: int = None):
    """Update TTL settings for caches"""
    global RESULTS_TTL, STREAM_INFO_TTL

    if results_ttl is not None and results_ttl > 0:
        RESULTS_TTL = results_ttl

    if stream_info_ttl is not None and stream_info_ttl > 0:
        STREAM_INFO_TTL = stream_info_ttl

    return {
        "results_ttl": RESULTS_TTL,
        "stream_info_ttl": STREAM_INFO_TTL
    }


@app.delete("/cache/results")
async def clear_results_cache():
    """Clear the results cache"""
    result_keys = valkey_client.keys("result:*")
    if result_keys:
        valkey_client.delete(*result_keys)

    return {"status": "ok", "cleared_keys": len(result_keys)}


@app.get("/health")
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

    # Overall status
    status = "healthy" if valkey_healthy and rabbitmq_healthy else "unhealthy"

    return {
        "status": status,
        "gateway_id": GATEWAY_ID,
        "valkey": "connected" if valkey_healthy else "disconnected",
        "rabbitmq": "connected" if rabbitmq_healthy else "disconnected",
        "timestamp": time.time()
    }


@app.delete("/streams/worker/{worker_id}")
async def clear_worker_streams(worker_id: str):
    """Clear all stream assignments for a specific worker (for manual recovery)"""
    try:
        # Get all stream location keys
        stream_location_keys = valkey_client.keys("stream_location:*")
        cleared_streams = []

        for key in stream_location_keys:
            stream_url = key.replace("stream_location:", "")
            assigned_worker = valkey_client.get(key)

            if assigned_worker == worker_id:
                logger.info(f"Clearing stream {stream_url} assignment from worker {worker_id}")
                valkey_client.delete(key)
                cleared_streams.append(stream_url)

        # Try to delete the worker record too
        valkey_client.delete(f"worker:{worker_id}")

        return {
            "status": "ok",
            "message": f"Cleared {len(cleared_streams)} streams assigned to worker {worker_id}",
            "cleared_streams": cleared_streams
        }

    except Exception as e:
        logger.error(f"Error clearing worker streams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear worker streams: {str(e)}")

@app.post("/streams/reassign")
async def reassign_stream(stream_url: str, new_worker_id: Optional[str] = None):
    """
    Manually reassign a stream to a different worker or clear its assignment.
    If new_worker_id is None, just clears the current assignment.
    """
    try:
        # Check if the stream exists
        stream_key = get_stream_location_key(stream_url)
        current_worker = valkey_client.get(stream_key)

        if not current_worker:
            return {
                "status": "ok",
                "message": f"Stream {stream_url} was not assigned to any worker"
            }

        # Clear the current assignment
        valkey_client.delete(stream_key)

        # Assign to new worker if specified
        if new_worker_id:
            # Check if the worker exists
            if is_worker_alive(new_worker_id):
                valkey_client.set(stream_key, new_worker_id, ex=STREAM_INFO_TTL)
                return {
                    "status": "ok",
                    "message": f"Reassigned stream {stream_url} from {current_worker} to {new_worker_id}"
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Cleared assignment for {stream_url}, but new worker {new_worker_id} is not active"
                }
        else:
            return {
                "status": "ok",
                "message": f"Cleared stream {stream_url} assignment from worker {current_worker}"
            }

    except Exception as e:
        logger.error(f"Error reassigning stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reassign stream: {str(e)}")

async def cleanup_dead_worker_streams():
    """
    Periodically scan for dead workers and clean up their stream assignments.
    This prevents requests from being permanently routed to dead workers.
    """
    while True:
        try:
            # Get all worker keys
            worker_keys = valkey_client.keys("worker:*")
            active_workers = set()

            for key in worker_keys:
                worker_id = key.replace("worker:", "")
                active_workers.add(worker_id)

            # Get all stream location keys
            stream_location_keys = valkey_client.keys("stream_location:*")

            for key in stream_location_keys:
                stream_url = key.replace("stream_location:", "")
                worker_id = valkey_client.get(key)

                if worker_id and worker_id not in active_workers:
                    logger.warning(f"Found stream {stream_url} assigned to dead worker {worker_id}. Cleaning up.")
                    valkey_client.delete(key)

            logger.info(f"Dead worker cleanup complete. Found {len(active_workers)} active workers.")

        except Exception as e:
            logger.error(f"Error during dead worker cleanup: {str(e)}")

        # Run every 2 minutes
        await asyncio.sleep(120)

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    # Initialize ValKey
    if not init_valkey():
        logger.error("Failed to initialize ValKey. API Gateway will have limited functionality.")

    # Initialize RabbitMQ
    if not await init_rabbitmq():
        logger.error("Failed to initialize RabbitMQ. API Gateway will have limited functionality.")

    # Start the dead worker cleanup task
    asyncio.create_task(cleanup_dead_worker_streams())

    logger.info("API Gateway startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    if rabbitmq_connection:
        await rabbitmq_connection.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )