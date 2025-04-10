# Transcript
import os
import math
import glob
import subprocess
import openai  # llm (whisper)
import streamlit as st
from pydub import AudioSegment

# Summary
from langchain.chat_models import ChatOpenAI  # llm (gpt)
from langchain.prompts import ChatPromptTemplate  # prompt
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser

# Q&A
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# Memory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from operator import itemgetter
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

streaming_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


# for cache (dev env)
has_transcript = os.path.exists("./.cache/podcast.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def invoke_chain(question):
    result = qa_chain.invoke(question)
    save_memory(question, result.content)


# summary cache
@st.cache_data(show_spinner=False)
def generate_summary(transcript_path):
    loader = TextLoader(transcript_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # st.write(len(docs)) # 25 -> 12 줄임

    # Chain
    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:
        """
    )

    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

    summary = first_summary_chain.invoke({"text": docs[0].page_content})

    refine_prompt = ChatPromptTemplate.from_template(
        """
        당신의 역할은 최종 요약본을 작성하는 것입니다.
        다음은 현재까지 생성된 요약입니다: {existing_summary}
        추가적인 문맥이 아래에 제공됩니다. 이 문맥이 유용하다고 판단될 경우, 이를 반영하여 기존 요약을 개선하세요.
        ------------
        {context}
        ------------
        새 문맥이 기존 요약을 더 나아지게 만들 수 없다면, 원래 요약을 그대로 유지하세요.
        """
    )

    refine_chain = refine_prompt | llm | StrOutputParser()

    with st.status("Summarizing...") as status:
        # execpt first summary
        for i, doc in enumerate(docs[1:]):
            status.update(label=f"Processing document {i+1}/{len(docs)-1}")
            # update(override) summary
            summary = refine_chain.invoke(
                {
                    "existing_summary": summary,
                    "context": doc.page_content,
                }
            )
            # st.write(summary) # 별 내용 없으면 요약이 없을 수도 있음.
    # final summary
    return summary


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


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

    audio_path = (
        video_path.replace(".mp4", ".mp3")
        .replace(".avi", ".mp3")
        .replace(".mkv", ".mp3")
        .replace(".mov", ".mp3")
    )
    # if os.path.exists(audio_path):
    #     return

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.set_page_config(page_title="MeetingGPT")

st.markdown(
    """
    # MettingGPT

    사이드바에서 동영상을 업로드하면 대화의 요약과 대화에 대한 질문을 할 수 있는 챗봇을 제공해 드립니다.
    """
)


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"

    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")

        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)

        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        if "isSummaryGenerated" not in st.session_state:
            st.session_state["isSummaryGenerated"] = False
        if "summary" not in st.session_state:
            st.session_state["summary"] = ""

        if st.button("Generate summary"):
            if st.session_state["isSummaryGenerated"]:
                st.write(st.session_state["summary"])
            else:
                summary = generate_summary(transcript_path)
                st.session_state["summary"] = summary
                st.session_state["isSummaryGenerated"] = True
                st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        question = st.text_input("회의에서 일어난 궁금한 일들을 물어보세요.")

        paint_history()

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    당신은 회의의 대본 기록을 이용해 사용자의 질문에 답변하는 전문적인 AI입니다.
                    당신이 기존에 알고 있던 지식을 사용하지 마세요.
                    모르는 내용이라면 모른다고 하고 지어내지 마세요.
                    사용자의 질문은 보통 회의에 대해 질문 하는 것이니 당신의 생각을 이야기 하지 마세요.
                    
                    대본: {context}
                    """,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        if question:
            send_message(question, "human")
            qa_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history"),
                }
                | qa_prompt
                | streaming_llm
            )

            with st.chat_message("ai"):
                invoke_chain(question)

else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
    )
    st.session_state["isSummaryGenerated"] = False
