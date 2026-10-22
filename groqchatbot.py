
import os
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv

# ----------------------------------
# Configuration
# ----------------------------------
load_dotenv()

st.set_page_config(
    page_title="ANSAR AI PDF Summarizer",
    page_icon="📄",
    layout="wide"
)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ----------------------------------
# Extract text from PDF
# ----------------------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(
                f"Page {page_number}:\n{text}"
            )

    return "\n\n".join(pages_text)


# ----------------------------------
# Split long documents into chunks
# ----------------------------------
def split_text(text, chunk_size=12000):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


# ----------------------------------
# Summarize text with Groq AI
# ----------------------------------
def summarize_pdf(text, client):
    chunks = split_text(text)
    chunk_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional document summarizer. "
                        "Summarize the supplied PDF text accurately. "
                        "Preserve important facts, key ideas, names, "
                        "dates, figures, and conclusions. "
                        "Do not invent information."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize section {index} of {len(chunks)} "
                        f"of this PDF:\n\n{chunk}"
                    )
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500
        )

        chunk_summaries.append(
            response.choices[0].message.content
        )

    # Combine section summaries into one final summary
    combined_text = "\n\n".join(chunk_summaries)

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a clear, well-organized final summary "
                    "of the document based on the provided section "
                    "summaries. Include the main topic, key points, "
                    "important facts, and conclusion. "
                    "Avoid repetition and do not invent facts."
                )
            },
            {
                "role": "user",
                "content": combined_text
            }
        ],
        temperature=0.2,
        max_completion_tokens=2500
    )

    return final_response.choices[0].message.content


# ----------------------------------
# Streamlit User Interface
# ----------------------------------
st.title("Ansar's AI PDF Summarizer")
st.write(
    "Upload a PDF and let Groq AI summarize its "
    "main ideas, important points, and conclusions."
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("AI Model:", MODEL)
    st.caption("Powered by Groq AI")

if not API_KEY:
    st.error(
        "Groq API key not found. Add GROQ_API_KEY "
        "to your .env file and restart the app."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("✨ Summarize PDF", type="primary"):
        try:
            with st.spinner("Reading your PDF..."):
                pdf_text = extract_pdf_text(uploaded_file)

            if not pdf_text.strip():
                st.error(
                    "No readable text found. This PDF may contain "
                    "scanned images. OCR is needed for scanned PDFs."
                )
                st.stop()

            client = Groq(api_key=API_KEY)

            with st.spinner(
                "Groq AI is analyzing your document. Please wait..."
            ):
                summary = summarize_pdf(pdf_text, client)

            st.subheader("📝 Your PDF Summary")
            st.markdown(summary)

            st.download_button(
                label="⬇️ Download Summary",
                data=summary,
                file_name="pdf_summary.txt",
                mime="text/plain"
            )

            with st.expander("View extracted PDF text"):
                st.text_area(
                    "Extracted content",
                    pdf_text,
                    height=300
                )

        except Exception as error:
            st.error(f"Something went wrong: {error}")

st.divider()
st.caption("AI PDF Summarizer | Python + Streamlit + Groq")
