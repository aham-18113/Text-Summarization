import streamlit as st
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# OpenAI API key
OPENAI_API_KEY = "api-key"


def generate_response(txt):
    """Generate a summary using OpenAI's API."""
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
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

# Text input
txt_input = st.text_area("Enter your text:", "", height=200)

# Submit button
if st.button("Summarize") and txt_input:
    with st.spinner("Generating summary..."):
        summary = generate_response(txt_input)
        st.success("Summary:")
        st.write(summary)
