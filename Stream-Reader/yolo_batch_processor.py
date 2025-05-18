import asyncio
import json
import logging
import time
import threading
import io
from PIL import Image
import numpy as np
from frame_grabber_analyzer import active_tasks_lock
from frame_grabber_analyzer import active_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = 16  # Target batch size for optimal GPU efficiency
MIN_BATCH_SIZE = 1  # Minimum batch size (process even 1 frame after timeout)
MAX_WAIT_TIME = 0.1  # 100ms maximum wait time
HIGH_PRIORITY_THRESHOLD = 8  # Process immediately if priority >= this value



# Batch item structure - store all relevant info together
class BatchItem:
    def __init__(self, message, stream_url, request_id, frame_data, priority=5, do_ocr=False):
        self.message = message  # Original RabbitMQ message
        self.stream_url = stream_url
        self.request_id = request_id
        self.frame_data = frame_data  # Actual image data
        self.priority = priority
        self.do_ocr = do_ocr  # Flag for OCR processing
        self.timestamp = time.time()
        self.image = None  # Will be lazily loaded when needed

    def get_image(self):
        """Get PIL Image from frame data (lazy loading)"""
        if self.image is None and self.frame_data:
            self.image = Image.open(io.BytesIO(self.frame_data))
        return self.image

    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority > other.priority  # Higher priority first


# Batch processing state
current_batch = []
batch_start_time = None
batch_lock = asyncio.Lock()
batch_event = asyncio.Event()


async def collect_message(message, frame):
    """Handle incoming message from RabbitMQ"""
    try:
        # Parse message
        request_data = json.loads(message.body.decode())
        stream_url = request_data.get("stream_url")
        request_id = request_data.get("request_id")
        priority = request_data.get("priority", 5)  # Default priority 5
        do_ocr = request_data.get("do_ocr", False)  # OCR flag

        # High priority messages skip batching
        if priority >= HIGH_PRIORITY_THRESHOLD:
            await process_single_message(message, stream_url, request_id, do_ocr)
            return

        # Get frame data
        try:
            frame_data = frame

            # Create batch item
            item = BatchItem(
                message=message,
                stream_url=stream_url,
                request_id=request_id,
                frame_data=frame_data,
                priority=priority,
                do_ocr=do_ocr
            )

            # Add to batch
            await add_to_batch(item)

        except Exception as e:
            logger.error(f"Error preparing frame: {str(e)}")
            await message.ack()  # Acknowledge failed message

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await message.ack()


async def add_to_batch(item):
    """Add item to the current processing batch"""
    global current_batch, batch_start_time

    async with batch_lock:
        # Start timer for first message in batch
        if not current_batch:
            batch_start_time = time.time()

        # Add to batch
        current_batch.append(item)

        # Signal processing if batch is full
        if len(current_batch) >= BATCH_SIZE:
            batch_event.set()


async def batch_processor_loop():
    """Main loop for processing batches"""
    global current_batch, batch_start_time

    while True:
        try:
            # Wait for signal or timeout
            try:
                # Check 4x per wait interval
                await asyncio.wait_for(batch_event.wait(), timeout=MAX_WAIT_TIME / 4)
                batch_event.clear()
            except asyncio.TimeoutError:
                # Timeout is normal, just continue to check conditions
                pass

            # Check if we should process the batch
            process_now = False
            current_size = 0
            elapsed_ms = 0

            async with batch_lock:
                current_size = len(current_batch)

                if current_size >= BATCH_SIZE:
                    # Batch is full - process immediately
                    process_now = True
                elif current_size >= MIN_BATCH_SIZE and batch_start_time:
                    # Check timeout condition
                    elapsed_ms = (time.time() - batch_start_time) * 1000
                    if elapsed_ms >= MAX_WAIT_TIME * 1000:
                        process_now = True

                # Get batch if ready to process
                if process_now and current_batch:
                    batch_to_process = current_batch.copy()
                    current_batch = []
                    batch_start_time = None
                else:
                    batch_to_process = []

            # Process outside the lock
            if batch_to_process:
                logger.info(f"Processing batch of {len(batch_to_process)} frames " +
                            f"(waited {elapsed_ms:.1f}ms)")
                await process_batch(batch_to_process)

        except Exception as e:
            logger.error(f"Error in batch processor: {str(e)}")
            await asyncio.sleep(0.5)  # Brief pause on error


async def process_batch(batch_items):
    """Process a batch of frames with single YOLO inference call"""
    try:
        # Track active tasks
        with active_tasks_lock:
            active_tasks += 1

        # Prepare batch for YOLO
        frames = []
        ocr_items = []  # Items that need OCR

        # Process each item
        for item in batch_items:
            try:
                # Get image from frame data
                image = item.get_image()
                if not image:
                    logger.error(f"Failed to get image for {item.stream_url}")
                    await item.message.ack()
                    continue

                # Add to frames list
                frames.append(image)

                # Check if OCR needed
                if item.do_ocr:
                    ocr_items.append(item)

                # Update status
                await publish_status_update(item.request_id, "processing")

            except Exception as e:
                logger.error(f"Error preparing item: {str(e)}")
                await item.message.ack()

        # Run YOLO inference on the batch
        if frames:
            # Single YOLO inference call for all frames
            results = model.predict(frames, imgsz=(736, 1280), device=device)

            # Process each result
            for i, result in enumerate(results):
                try:
                    item = batch_items[i]

                    # Extract detections
                    detections = []
                    for box in result.boxes:
                        obj = {
                            "class": result.names[int(box.cls.item())],
                            "confidence": float(box.conf.item()),
                            "bbox": [float(x) for x in box.xyxy[0]]
                        }
                        detections.append(obj)

                    # Prepare result
                    analysis_result = {
                        "stream_url": item.stream_url,
                        "timestamp": time.time(),
                        "detections": detections,
                        "frame_width": frames[i].width,
                        "frame_height": frames[i].height,
                        "worker_id": WORKER_ID,
                        "batch_processed": True,
                        "batch_size": len(frames),
                        "ocr_requested": item.do_ocr
                    }

                    # Process OCR if needed
                    if item.do_ocr:
                        try:
                            # Call OCR processing function (implement this separately)
                            ocr_results = await process_ocr(frames[i], detections)
                            analysis_result["ocr_results"] = ocr_results
                        except Exception as e:
                            logger.error(f"OCR processing error: {str(e)}")
                            analysis_result["ocr_error"] = str(e)

                    # Send result
                    await publish_status_update(
                        item.request_id,
                        "completed",
                        result=analysis_result
                    )

                    # Acknowledge
                    await item.message.ack()

                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    await batch_items[i].message.ack()

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        for item in batch_items:
            await item.message.ack()
    finally:
        with active_tasks_lock:
            active_tasks -= 1


async def process_single_message(message, stream_url, request_id, do_ocr=False):
    """Process high-priority message immediately (no batching)"""
    try:
        with active_tasks_lock:
            active_tasks += 1

        # Get frame
        stream_ctx = initialize_stream(stream_url)
        frame_data = grab_frame(stream_ctx)
        image = Image.open(io.BytesIO(frame_data))

        # Update status
        await publish_status_update(request_id, "processing")

        # Run inference (single frame)
        results = model.predict(image, imgsz=(736, 1280), device=device)

        # Process result
        detections = []
        for box in results[0].boxes:
            obj = {
                "class": results[0].names[int(box.cls.item())],
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
            "worker_id": WORKER_ID,
            "batch_processed": False,
            "priority_processed": True,
            "ocr_requested": do_ocr
        }

        # Process OCR if needed
        if do_ocr:
            try:
                # Call OCR processing function
                ocr_results = await process_ocr(image, detections)
                analysis_result["ocr_results"] = ocr_results
            except Exception as e:
                logger.error(f"OCR processing error: {str(e)}")
                analysis_result["ocr_error"] = str(e)

        # Send result
        await publish_status_update(
            request_id,
            "completed",
            result=analysis_result
        )

        # Acknowledge
        await message.ack()

    except Exception as e:
        logger.error(f"Error processing high-priority message: {str(e)}")
        await message.ack()
    finally:
        with active_tasks_lock:
            active_tasks -= 1


# Placeholder for OCR function - implement this based on your OCR needs
async def process_ocr(image, detections):
    """Process OCR on detected text regions"""
    # This is a placeholder - implement your actual OCR logic
    # You could use something like pytesseract, easyocr, or a custom model

    # Example implementation:
    # import pytesseract
    # text_results = {}
    #
    # for i, detection in enumerate(detections):
    #     if detection["class"] == "text":  # If your YOLO model detects text regions
    #         bbox = detection["bbox"]
    #         x1, y1, x2, y2 = [int(v) for v in bbox]
    #         text_region = image.crop((x1, y1, x2, y2))
    #         text = pytesseract.image_to_string(text_region)
    #         text_results[f"text_{i}"] = {
    #             "bbox": bbox,
    #             "text": text,
    #             "confidence": detection["confidence"]
    #         }
    #
    # return text_results

    # For now, return placeholder
    return {"status": "OCR processing not implemented"}


async def init_rabbitmq():
    """Initialize RabbitMQ with batch processing"""
    global rabbitmq_connection, rabbitmq_channel, batch_processor_task

    try:
        # Connect to RabbitMQ
        rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
        rabbitmq_channel = await rabbitmq_connection.channel()

        # Set prefetch to ensure we get enough messages for batching
        await rabbitmq_channel.set_qos(prefetch_count=BATCH_SIZE * 2)

        # Set up exchanges and queues (as in your original code)
        analysis_exchange = await rabbitmq_channel.declare_exchange(
            "analysis_exchange",
            aio_pika.ExchangeType.DIRECT
        )

        status_exchange = await rabbitmq_channel.declare_exchange(
            "status_exchange",
            aio_pika.ExchangeType.TOPIC
        )

        analysis_queue = await rabbitmq_channel.declare_queue(
            "analysis_queue",
            durable=True,
            arguments={"x-max-priority": 5}
        )

        # Bind queue to exchange
        await analysis_queue.bind(analysis_exchange, routing_key="analysis")

        # Start consuming
        await analysis_queue.consume(collect_message)

        # Start batch processor
        batch_processor_task = asyncio.create_task(batch_processor_loop())

        logger.info(f"RabbitMQ initialized with batch processing (size={BATCH_SIZE}, timeout={MAX_WAIT_TIME}s)")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ: {str(e)}")
        return False