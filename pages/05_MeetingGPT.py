import os
import math
import glob
import subprocess
import openai
import streamlit as st
from pydub import AudioSegment

# for cache (dev env)
has_transcript = os.path.exists("./.cache/podcast.txt")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return

    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()  # in-place 정렬
    # files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return

    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",  # override -y flag
        "-i",  # input
        video_path,
        "-vn",  # ingnore video
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunks_size, chunks_folder):
    if has_transcript:
        return

    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunks_size * 60 * 1000  # milliseconds
    chunks = math.ceil(len(track) / chunk_len)

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]

        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")


st.set_page_config(page_title="MeetingGPT")

st.markdown(
    """
    # MettingGPT

    Upload a video then give you a transcript, a summary and a chat bot to ask any questions about it.

    Get started by uploading a video file in the sidebar.
    """
)


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"

    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")

        with open(video_path, "wb") as f:
            f.write(video_content)

    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)

    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunks_folder)

    with st.status("Transcribing audio..."):
        transcribe_chunks(chunks_folder, transcript_path)
