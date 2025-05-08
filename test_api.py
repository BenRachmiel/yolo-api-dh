import requests
import argparse
import time
import urllib.parse
from PIL import Image
import io
import matplotlib.pyplot as plt


def test_frame_grabber(service_url, hls_url, num_frames=5, interval=1):
    """
    Test the frame grabber service by requesting multiple frames

    Args:
        service_url: URL of the frame grabber service (e.g., http://localhost:8000)
        hls_url: URL of the HLS stream to grab frames from
        num_frames: Number of frames to grab
        interval: Time interval between frame grabs in seconds
    """
    # Encode the HLS URL for the API path
    encoded_url = urllib.parse.quote(hls_url, safe='')

    # Replace :// with ___ as implemented in the service
    encoded_url = encoded_url.replace("%3A%2F%2F", "___")

    # Create the request URL
    frame_url = f"{service_url}/frame/{encoded_url}"

    print(f"Testing frame grabber at: {service_url}")
    print(f"HLS stream URL: {hls_url}")
    print(f"Grabbing {num_frames} frames with {interval}s interval")
    print("-" * 50)

    # Test first frame timing (cold start)
    start_time = time.time()
    response = requests.get(frame_url)
    cold_start_time = time.time() - start_time

    if response.status_code != 200:
        print(f"Error requesting frame: {response.status_code}")
        print(response.text)
        return

    print(f"Cold start frame grab time: {cold_start_time:.4f} seconds")

    # Save first frame
    img = Image.open(io.BytesIO(response.content))
    img.save("frame_0.jpg")

    # Grab remaining frames
    times = []
    for i in range(1, num_frames):
        time.sleep(interval)

        start_time = time.time()
        response = requests.get(frame_url)
        request_time = time.time() - start_time
        times.append(request_time)

        if response.status_code != 200:
            print(f"Error requesting frame {i}: {response.status_code}")
            continue

        # Save frame
        img = Image.open(io.BytesIO(response.content))
        img.save(f"frame_{i}.jpg")

        print(f"Frame {i} grab time: {request_time:.4f} seconds")

    # Display statistics
    if times:
        print("-" * 50)
        print(f"Avg subsequent frame grab time: {sum(times) / len(times):.4f} seconds")
        print(f"Min frame grab time: {min(times):.4f} seconds")
        print(f"Max frame grab time: {max(times):.4f} seconds")

    # Check all active streams
    print("-" * 50)
    print("Checking active streams:")
    response = requests.get(f"{service_url}/streams")
    if response.status_code == 200:
        streams = response.json()
        print(f"Total active streams: {streams['total']}")
        for idx, stream in enumerate(streams['streams']):
            print(f"Stream {idx + 1}: {stream['url']} (Status: {stream['status']})")
    else:
        print("Error checking streams")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the HLS Frame Grabber service")
    parser.add_argument("--service", default="http://localhost:8000", help="Frame grabber service URL")
    parser.add_argument("--stream", required=True, help="HLS stream URL to test")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to grab")
    parser.add_argument("--interval", type=float, default=1.0, help="Time between frames (seconds)")

    args = parser.parse_args()

    test_frame_grabber(args.service, args.stream, args.frames, args.interval)