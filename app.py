
import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


def extract_youtube_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|$)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "ar"])
    return " ".join([entry["text"] for entry in transcript])

def split_transcript(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.from_documents(chunks, embeddings)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_messages(inputs):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer ONLY based on the transcript. If it's insufficient, say: I don't know."
        },
        {
            "role": "user",
            "content": f"question: {inputs['question']}\n\ncontext:\n{inputs['context']}"
        }
    ]


client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="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"

)
zephyr_chat = RunnableLambda(lambda messages: client.chat_completion(messages=messages, max_tokens=512))
parser = RunnableLambda(lambda x: x.get('content') if isinstance(x, dict) else str(x))


st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title(" Chat with a YouTube Video")

url = st.text_input("🔗 Enter YouTube URL:")
if url:
    video_id = extract_youtube_id(url)
    if video_id:
        try:
            transcript = get_transcript(video_id)
            chunks = split_transcript(transcript)
            vectorstore = build_vectorstore(chunks)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            chain = parallel_chain | RunnableLambda(build_messages) | zephyr_chat | parser

            st.success("✅ Transcript processed. Ready to chat!")

            query = st.text_input("💬 Ask your question about the video:")
            if query:
                with st.spinner("Thinking..."):
                    answer = chain.invoke(query)
                    st.markdown(f"**Answer:** {answer}")

        except TranscriptsDisabled:
            st.error("❌ Transcript is not available for this video.")
    else:
        st.warning("⚠️ Invalid YouTube URL.")
