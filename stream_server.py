import os
import argparse
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import ffmpeg
import subprocess
import signal


def get_local_ip():
    """Get the local IP address of the machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def start_ffmpeg_stream(input_file, output_dir, duration=None):
    """Start FFmpeg process to create HLS stream using ffmpeg-python"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build FFmpeg command using ffmpeg-python
    stream = ffmpeg.input(input_file, stream_loop=-1, re=None)

    # Add duration if specified
    if duration:
        stream = stream.filter('trim', duration=duration)

    # Setup output options
    output = stream.output(
        f'{output_dir}/playlist.m3u8',
        codec='copy',  # Use copy to maintain original quality
        f='hls',  # HLS format
        hls_time=4,  # Segment duration in seconds
        hls_list_size=5,  # Number of segments in playlist
        hls_flags='delete_segments',  # Delete old segments
        hls_segment_filename=f'{output_dir}/segment_%03d.ts'
    )

    # Setup video codec if not using copy
    # output = stream.output(
    #     f'{output_dir}/playlist.m3u8',
    #     vcodec='libx264', crf=21, preset='veryfast',
    #     acodec='aac', audio_bitrate='128k',
    #     f='hls',
    #     hls_time=4,
    #     hls_list_size=5,
    #     hls_flags='delete_segments',
    #     hls_segment_filename=f'{output_dir}/segment_%03d.ts'
    # )

    # Get the command for logging purposes
    cmd = ffmpeg.compile(output, overwrite_output=True)
    print(f"Starting FFmpeg with command: {' '.join(cmd)}")

    # Start the FFmpeg process
    process = output.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, overwrite_output=True)

    print(f"Started FFmpeg process to create HLS stream from '{input_file}'")
    print(f"Output directory: {output_dir}")

    return process


def start_http_server(directory, port):
    """Start HTTP server to serve HLS content"""
    # Change to the specified directory
    os.chdir(directory)

    # Create and start HTTP server
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    local_ip = get_local_ip()
    print(f"\nHTTP server started on port {port}")
    print(f"Your HLS stream is available at: http://{local_ip}:{port}/playlist.m3u8")
    print(f"Local URL: http://localhost:{port}/playlist.m3u8")
    print("Press Ctrl+C to stop the server and streaming\n")

    return server


def check_file_exists(file_path):
    """Check if a file exists and is a valid media file"""
    if not os.path.isfile(file_path):
        return False

    try:
        # Try to get information about the file using ffprobe
        probe = ffmpeg.probe(file_path)
        # Check if there are streams in the file
        if 'streams' in probe and len(probe['streams']) > 0:
            return True
        return False
    except ffmpeg.Error:
        return False


def main():
    parser = argparse.ArgumentParser(description="Create an HLS stream from a video file using ffmpeg-python")
    parser.add_argument("input_file", help="Input video file")
    parser.add_argument("--output", "-o", default="./hls_output", help="Output directory")
    parser.add_argument("--port", "-p", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--duration", "-d", type=int, help="Duration in seconds (optional)")

    args = parser.parse_args()

    # Validate input file
    if not check_file_exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found or is not a valid media file!")
        return 1

    # Get absolute path for the output directory
    output_dir = os.path.abspath(args.output)

    try:
        # Start FFmpeg process
        ffmpeg_process = start_ffmpeg_stream(args.input_file, output_dir, args.duration)

        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down gracefully...")
            if ffmpeg_process:
                ffmpeg_process.terminate()
                try:
                    ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ffmpeg_process.kill()
            print("Stopped.")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Allow some time for FFmpeg to create initial segments
        print("Waiting for FFmpeg to generate initial segments...")
        time.sleep(10)

        # Start HTTP server
        http_server = start_http_server(output_dir, args.port)

        # Keep the main thread alive
        while True:
            time.sleep(1)
            if ffmpeg_process.poll() is not None:
                print("FFmpeg process exited unexpectedly!")
                break

    except KeyboardInterrupt:
        print("\nStopping server and FFmpeg...")
        if 'ffmpeg_process' in locals() and ffmpeg_process:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()

    except Exception as e:
        print(f"Error: {e}")
        if 'ffmpeg_process' in locals() and ffmpeg_process:
            ffmpeg_process.terminate()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())