import argparse
import requests
import time
import os

def upload_and_download(video_path, output_filename, endpoint_url):
    if not os.path.isfile(video_path):
        print(f"❌ Error: File not found — {video_path}")
        return

    print(f"📤 Uploading '{video_path}' to server...")

    with open(video_path, 'rb') as f:
        files = {'video': f}
        try:
            response = requests.post(endpoint_url, files=files)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Upload failed: {e}")
            return

    data = response.json()
    captioned_url = data.get("captionedVideoUrl")

    if not captioned_url:
        print(f"❌ No captioned video URL returned. Server response:\n{data}")
        return

    print(f"✅ Captioned video ready: {captioned_url}")
    print(f"⬇️ Downloading to '{output_filename}'...")

    try:
        r = requests.get(captioned_url, stream=True)
        r.raise_for_status()
        with open(output_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return

    print(f"🎉 Saved captioned video to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a video to the captioning server and download the result."
    )
    parser.add_argument("video", help="Path to the input video file (e.g., myclip.mp4)")
    parser.add_argument(
        "--output",
        help="Output filename for the captioned video (default: captioned_output.mp4)",
        default="captioned_output.mp4"
    )
    parser.add_argument(
        "--url",
        help="Server endpoint URL (default: https://ai-content-pipeline.onrender.com/caption-video)",
        default="https://ai-content-pipeline.onrender.com/caption-video"
    )

    args = parser.parse_args()

    upload_and_download(args.video, args.output, args.url)
