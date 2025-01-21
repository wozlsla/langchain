from unittest import result
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore

# from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


answer_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answer_chain = answer_prompt | llm

    return {
        "question": question,
        "answer": [
            {
                "answer": answer_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the 'Answer' to answers that have the highest 'Score' (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.
            
            Choose the most informed answer among the answers with the same score.

            You should always respond to the source.

            Answers: {answers}
            ---
            Examples:
                                                  
            The moon is 384,400 km away.

            Source: https://example.com
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answer"]
    question = inputs["question"]
    # choose_chain = choose_prompt | llm(streaming=True) X
    choose_chain = choose_prompt | llm_with_streaming
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    # st.write(condensed)
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        # .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        # filter_urls=[r"^(?!.*\/blog\/).*"],
        filter_urls=[
            r"(.*\/ai-gateway\/).*",
            r"(.*\/vectorize\/).*",
            r"(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # caching
    url_copy = url[:]
    cache_filename = url_copy.replace("/", "_")
    cache_filename.strip()
    cache_dir = LocalFileStore(f"./.cache/{cache_filename}/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )

    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="./files/ham.ico",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")

    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
        disabled=True,
    )
    st.markdown("---")


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        paint_history()

        llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
        )

        llm_with_streaming = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )

        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")

        if query:
            send_message(query, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            with st.chat_message("ai"):
                chain.invoke(query)
