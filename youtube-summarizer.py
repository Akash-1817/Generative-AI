#!pip install langchain transformers yt-dlp pydub
#!pip install -U langchain-community
#!pip install youtube-transcript-api

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# Step 1: Get YouTube Transcript
def fetch_transcript(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    docs = loader.load()
    if not docs:
        raise ValueError("❌ No transcript found.")
    return docs[0].page_content

# Step 2: Split Transcript
def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Step 3: Translate Tamil → English (if needed)
def translate_chunks(chunks, source_lang='ta'):
    if source_lang == 'ta':
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ta")
        translated = []
        for chunk in chunks:
            try:
                result = translator(chunk[:512])[0]['translation_text']
                translated.append(result)
            except Exception as e:
                print(f"⚠️ Translation error: {e}")
        return translated
    return chunks

# Step 4: Summarize Text
def summarize_chunks(chunks):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=50, min_length=10, do_sample=False)
            summaries.append(result[0]['summary_text'])
        except Exception as e:
            print(f"⚠️ Summarization error: {e}")
    return "\n".join(summaries)

# Step 5: Full Pipeline
def summarize_youtube_video(video_url, source_language='auto'):
    print("📥 Fetching transcript...")
    transcript = fetch_transcript(video_url)

    print("✂️ Splitting transcript...")
    chunks = split_text(transcript)

    if source_language == 'ta':
        print("🌐 Translating Tamil → English...")
        chunks = translate_chunks(chunks, source_lang='ta')

    print("🧠 Summarizing chunks...")
    final_summary = summarize_chunks(chunks)

    return final_summary

# Run
if __name__ == "__main__":
    url = input("📺 YouTube URL: ").strip()
    lang = input("🌐 Language? ('ta' for Tamil, leave empty for English): ").strip() or 'auto'
    try:
        summary = summarize_youtube_video(url, source_language=lang)
        print("\n✅ Final Summary:\n")
        print(summary)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

