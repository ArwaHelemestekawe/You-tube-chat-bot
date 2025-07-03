# Youtube-chat-bot



#  YouTube RAG Chatbot with LangChain + Zephyr

This is a Streamlit app that lets you **chat with any YouTube video** using **Retrieval-Augmented Generation (RAG)** and Hugging Face's **Zephyr-7B** model.

You paste a YouTube URL, the app:
- extracts the video ID
- grabs its transcript (via `youtube_transcript_api`)
- splits it into chunks
- embeds it with `all-MiniLM-L6-v2`
- stores it in a FAISS vector store
- answers your questions by retrieving the most relevant transcript snippets and sending them to Zephyr

---

## 🚀 Features

-  Accepts YouTube video links
-  Extracts + embeds transcripts automatically
-  Uses FAISS similarity search
-  Answers using `HuggingFaceH4/zephyr-7b-beta`
-  Interactive chat interface built with Streamlit

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit langchain langchain-community langchain-huggingface huggingface-hub sentence-transformers youtube-transcript-api
```


---

## 🔐 Hugging Face Token

Set your token inside the script:

```python
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token="hf_************"
)
```

Or manage it securely via environment variables / secrets.

---

## 🏁 Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
 project/
┣  app.py
┣  requirements.txt
┣  README.md
```

---

## ✅ Example Models

- Embedding: `all-MiniLM-L6-v2`
- LLM: `HuggingFaceH4/zephyr-7b-beta`

---

##  Notes

- Only works with videos that have open captions (transcripts).
- You can customize the `system` prompt or use your own LLM.
- Streamlit makes it easy to prototype, but you can productionize this using FastAPI or Gradio.

---


## 🙌 Acknowledgments

Built using:
- [LangChain](https://www.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
```

---

