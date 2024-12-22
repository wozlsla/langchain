from docx import document
from nltk.chat import Chat
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st


st.set_page_config(
    page_title="QuizGPT",
    page_icon="./files/ham.ico",
)

st.title("Quiz GPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# For question, 다양하게 구성
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
                
            Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
            
            Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
            Use (o) to signal the correct answer.
                
            Question examples:
                
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
                
            Question: What is the capital or Georgia?
            Answers: Baku|Tbilisi(o)|Manila|Beirut
                
            Question: When was Avatar released?
            Answers: 2007|2001|2009(o)|1998
                
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o)|Painter|Actor|Model
                
            If context language is Korean, you should make it by Korean.
            Your turn!
                            
            Context: {context}
        """,
        )
    ]
)

questions_chain = {"context": format_docs} | question_prompt | llm


# For formatting
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf file.",
            type=["pdf", "docx", "txt"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            # retriever = WikipediaRetriever(top_k_results=5)
            retriever = WikipediaRetriever(top_k_results=5, lang="ko")
            with st.status("Searching..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:

    start = st.button("Generate Quiz")

    if start:
        questions_response = questions_chain.invoke(docs)
        st.write(questions_response.content)
        formatting_response = formatting_chain.invoke(
            {"context": questions_response.content},
        )
        st.write(formatting_response.content)
