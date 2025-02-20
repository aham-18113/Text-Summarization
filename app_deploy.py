import streamlit as st
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


def generate_response(txt, openai_api_key):
    """Generate a summary using OpenAI's API."""
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)


# Page title
st.set_page_config(page_title="🦜🔗 Text Summarization App")
st.title("🦜🔗 Text Summarization App")

# OpenAI API Key input
openai_api_key = st.text_input("Enter OpenAI API key:", type="password")

# Text input
txt_input = st.text_area("Enter your text:", "", height=200)

# Submit button
if st.button("Summarize") and openai_api_key.startswith("sk-") and txt_input:
    with st.spinner("Generating summary..."):
        summary = generate_response(txt_input, openai_api_key)
        st.success("Summary:")
        st.write(summary)

# Instructions for getting an OpenAI API key
st.subheader("Get an OpenAI API key:")
st.write(
    """
1. Go to [OpenAI API keys](https://platform.openai.com/account/api-keys).
2. Click on the `+ Create new secret key` button.
3. Copy and paste the key above to use the summarization app.
"""
)
