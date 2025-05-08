import argparse
import asyncio
import time
import logging
from typing import Dict, Any
import httpx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_GATEWAY_URL = "http://192.168.1.102:10800"
DEFAULT_STREAM_URL = "https://content.jwplatform.com/manifests/vM7nH0Kl.m3u8?abcde"
DEFAULT_TEST_COUNT = 40
DEFAULT_DELAY = 0.5  # seconds between tests, to stay within 10s window


async def test_api_caching(
        gateway_url: str,
        stream_url: str,
        test_count: int,
        delay: float,
        force_refresh: bool = False
) -> None:
    """
    Test the API gateway caching by making multiple requests for the same stream
    within a short time window.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        request_ids = []
        start_time = time.time()

        # Make initial request
        logger.info(f"Starting test with stream URL: {stream_url}")
        logger.info(f"Will make {test_count} requests with {delay}s delay between them")

        # Run multiple tests
        for i in range(test_count):
            test_start = time.time()
            logger.info(f"\n--- Test #{i + 1} ---")

            # 1. Request analysis
            analyze_response = await request_analysis(client, gateway_url, stream_url, force_refresh)
            request_id = analyze_response.get("request_id")
            request_ids.append(request_id)

            if not request_id:
                logger.error(f"Failed to get request_id from response: {analyze_response}")
                continue

            # 2. Check if result was from cache
            if analyze_response.get("status") == "completed":
                logger.info(f"✅ CACHE HIT! Request returned completed status immediately")
                logger.info(f"Result: {analyze_response.get('result', {}).get('detections', [])[:3]}...")
            else:
                logger.info(f"⏳ Cache miss. Request {request_id} status: {analyze_response.get('status')}")

                # 3. If not completed, poll for result
                result = await poll_for_result(client, gateway_url, request_id)
                if result:
                    logger.info(f"✅ Got result after polling: {result.get('status')}")
                    detections = result.get('result', {}).get('detections', [])
                    logger.info(f"Detections found: {len(detections)}")
                    if detections:
                        logger.info(f"Sample: {detections[:3]}...")
                else:
                    logger.error(f"❌ Failed to get result for request {request_id}")

            test_duration = time.time() - test_start
            logger.info(f"Test #{i + 1} completed in {test_duration:.2f}s")

            # Wait a bit before the next request, but ensure we don't exceed 10s total
            elapsed = time.time() - start_time
            if i < test_count - 1:  # Don't delay after the last test
                if elapsed + delay > 10.0:
                    logger.info(f"Reducing delay to stay within 10s window (elapsed: {elapsed:.2f}s)")
                    await asyncio.sleep(max(0.5, 10.0 - elapsed))
                else:
                    await asyncio.sleep(delay)

        # Summary
        total_duration = time.time() - start_time
        logger.info(f"\n=== Test Summary ===")
        logger.info(f"Completed {test_count} requests in {total_duration:.2f}s")
        logger.info(f"Average time per request: {total_duration / test_count:.2f}s")
        logger.info(f"Request IDs: {request_ids}")


async def request_analysis(
        client: httpx.AsyncClient,
        gateway_url: str,
        stream_url: str,
        force_refresh: bool
) -> Dict[str, Any]:
    """Make a request to the analyze endpoint and return the response."""
    start_time = time.time()
    logger.info(f"Requesting analysis of {stream_url} (force_refresh={force_refresh})")

    try:
        response = await client.post(
            f"{gateway_url}/analyze",
            json={
                "url": stream_url,
                "force_refresh": force_refresh,
                "priority": 5  # Use high priority for faster processing
            }
        )

        if response.status_code != 200:
            logger.error(f"Request failed with status {response.status_code}: {response.text}")
            return {}

        result = response.json()
        logger.info(
            f"Analysis request completed in {time.time() - start_time:.2f}s with status: {result.get('status')}")
        return result

    except Exception as e:
        logger.error(f"Error requesting analysis: {str(e)}")
        return {}


async def poll_for_result(
        client: httpx.AsyncClient,
        gateway_url: str,
        request_id: str,
        max_attempts: int = 5,
        poll_interval: float = 0.5
) -> Dict[str, Any]:
    """Poll the result endpoint until the analysis is completed or max attempts reached."""
    logger.info(f"Polling for result of request {request_id}")

    for attempt in range(max_attempts):
        try:
            response = await client.get(f"{gateway_url}/analyze/{request_id}")

            if response.status_code != 200:
                logger.error(f"Poll request failed with status {response.status_code}: {response.text}")
                await asyncio.sleep(poll_interval)
                continue

            result = response.json()
            status = result.get("status")

            if status == "completed":
                logger.info(f"Request {request_id} completed on poll attempt {attempt + 1}")
                return result
            elif status == "failed":
                logger.error(f"Request {request_id} failed: {result.get('error', 'Unknown error')}")
                return result
            else:
                logger.info(f"Request {request_id} status: {status}, waiting...")
                await asyncio.sleep(poll_interval)

        except Exception as e:
            logger.error(f"Error polling for result: {str(e)}")
            await asyncio.sleep(poll_interval)

    logger.warning(f"Max poll attempts reached for request {request_id}")
    return {}


async def check_valkey_cache(redis_url: str, stream_url: str) -> None:
    """Directly check the Valkey cache for the given stream URL (for debugging)."""
    try:
        import redis
        # Parse Redis URL format: redis://:password@host:port/db
        # Example: redis://:dev_password@localhost:6379/0
        parts = redis_url.split('@')
        if len(parts) > 1:
            auth_part = parts[0].replace('redis://', '')
            conn_part = parts[1]
        else:
            auth_part = ''
            conn_part = parts[0].replace('redis://', '')

        if ':' in auth_part:
            _, password = auth_part.split(':')
        else:
            password = None

        host_port = conn_part.split('/')[0]
        if ':' in host_port:
            host, port = host_port.split(':')
            port = int(port)
        else:
            host = host_port
            port = 6379

        # Connect to Redis
        client = redis.Redis(host=host, port=port, password=password, decode_responses=True)

        # Check cache key
        cache_key = f"result:{stream_url}"
        value = client.get(cache_key)

        if value:
            logger.info(f"Cache entry found for {cache_key}")
            logger.info(f"Value: {value[:100]}...")  # Show the first 100 chars

            # Get TTL
            ttl = client.ttl(cache_key)
            logger.info(f"TTL: {ttl}s")
        else:
            logger.info(f"No cache entry found for {cache_key}")

    except Exception as e:
        logger.error(f"Error checking Valkey cache: {str(e)}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test API Gateway caching')
    parser.add_argument('--gateway-url', type=str, default=DEFAULT_GATEWAY_URL,
                        help=f'API Gateway URL (default: {DEFAULT_GATEWAY_URL})')
    parser.add_argument('--stream-url', type=str, default=DEFAULT_STREAM_URL,
                        help=f'Stream URL to test (default: {DEFAULT_STREAM_URL})')
    parser.add_argument('--count', type=int, default=DEFAULT_TEST_COUNT,
                        help=f'Number of test requests to make (default: {DEFAULT_TEST_COUNT})')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help=f'Delay between requests in seconds (default: {DEFAULT_DELAY})')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh on the first request to ensure cache is populated')
    parser.add_argument('--redis-url', type=str,
                        help='Optional Redis URL to directly check cache (format: redis://:password@host:port/db)')
    return parser.parse_args()


async def main():
    """Main function to run the test."""
    args = parse_arguments()

    # First test - force refresh the first request to ensure cache is populated
    if args.force_refresh:
        logger.info("\n=== Running first test with force_refresh=True to populate cache ===")
        await test_api_caching(
            gateway_url=args.gateway_url,
            stream_url=args.stream_url,
            test_count=1,
            delay=0,
            force_refresh=True
        )

    # Main test - all requests within 10s window
    logger.info("\n=== Running main test with normal requests ===")
    await test_api_caching(
        gateway_url=args.gateway_url,
        stream_url=args.stream_url,
        test_count=args.count,
        delay=args.delay,
        force_refresh=False
    )

    # If Redis URL provided, check cache directly
    if args.redis_url:
        logger.info("\n=== Checking Valkey cache directly ===")
        await check_valkey_cache(args.redis_url, args.stream_url)


if __name__ == "__main__":
    asyncio.run(main())