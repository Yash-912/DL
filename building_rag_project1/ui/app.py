import streamlit as st
import requests
import os

st.set_page_config(page_title="Knowledge Base (RAG MVP)", layout="wide")

API_BASE_URL = "http://localhost:8000"

st.title("Enterprise Knowledge Base")
st.markdown("Ask questions about your documents.")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF, PPTX, or Markdown file", type=["pdf", "pptx", "md", "txt"])
    
    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Ingesting document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"{data['message']} ({data.get('chunks_indexed', 0)} chunks indexed)")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
        else:
            st.warning("Please select a file first.")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query", 
                    json={"query": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    message_placeholder.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.write(f"- {source}")
                                
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
