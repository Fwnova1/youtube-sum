import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import google.generativeai as genai
import yt_dlp
import openai
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# 🔑 Configure API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_google_gemini_api_key")  # Replace with your actual key if not using env
openai.api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")  # Replace with your actual key if not using env

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def summarizeyt_with_gemini(text, target_language):
    """Summarize text using Google Gemini API in the user-selected language."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Summarize this text so the user can understand the content of the video. Note down the important details. Be as natural as possible. The summary should be in {target_language}:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text


def extract_video_id(url):
    """Extracts the YouTube video ID from a given URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]

    match = re.match(r"(?:https?://)?(?:www\.)?youtu\.be/([^?&]+)", url)
    return match.group(1) if match else None


def fetch_transcript(video_url):
    """Auto-detect and fetch the best available YouTube transcript."""
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL."

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        print(f"DEBUG: Available transcripts for {video_id}:")
        for t in transcripts:
            print(f" - {t.language_code} (Generated: {t.is_generated})")

        # Prioritize manually created captions, fallback to auto-generated
        best_transcript = next(
            (t for t in transcripts if not t.is_generated),
            next((t for t in transcripts if t.is_generated), None)
        )

        if not best_transcript:
            raise TranscriptsDisabled

        transcript = best_transcript.fetch()
        transcript_text = " ".join([t["text"] for t in transcript])
        return transcript_text, None

    except TranscriptsDisabled:
        print("Captions are disabled. Using Whisper STT.")
        audio_path = download_audio(video_url)
        transcript_text = transcribe_audio(audio_path)
        return transcript_text, None

    except Exception as e:
        return None, f"Error: {str(e)}"


def download_audio(video_url):
    """Download audio from a YouTube video."""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "youtube_audio.mp3")

    print(f"Downloading audio to: {output_path}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
        except Exception as e:
            print(f"yt-dlp download failed: {e}")
            return None

    if not os.path.exists(output_path):
        print(f"Error: Audio file not found at {output_path}")
        return None

    return output_path


def transcribe_audio(audio_path):
    """
    Converts audio to text using OpenAI's updated API.
    """
    with open(audio_path, "rb") as audio_file:
        result = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format='text'
        )
    return result["text"]


def main():
    """Main function to fetch a YouTube transcript, transcribe audio if necessary, and summarize content."""
    video_url = input("Enter the YouTube video URL: ").strip()
    target_language = input("Enter the language for the summary (e.g., English, Vietnamese): ").strip()

    print("Fetching transcript...")
    transcript_text, error = fetch_transcript(video_url)

    if error:
        print(f"Error: {error}")
        return

    print("Transcript successfully fetched.")

    print("Summarizing with Google Gemini...")
    try:
        summary = summarizeyt_with_gemini(transcript_text, target_language)
        print("\n=== Video Summary ===")
        print(summary)
    except Exception as e:
        print(f"Error during summarization: {str(e)}")


if __name__ == "__main__":
    main()
