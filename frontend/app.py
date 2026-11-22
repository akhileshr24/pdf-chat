
import requests
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide",
)


# Conversation management

def create_conversation():
    response = requests.post(
        f"{API_URL}/conversations",
        timeout=10,
    )
    response.raise_for_status()

    return response.json()["conversation_id"]


def load_conversations():
    response = requests.get(
        f"{API_URL}/conversations",
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


def load_messages(conversation_id):
    response = requests.get(
        f"{API_URL}/conversations/{conversation_id}/messages",
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


def load_documents(conversation_id):
    response = requests.get(
        f"{API_URL}/conversations/{conversation_id}/documents",
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


# Validate the conversation ID and create a fresh chat if necessary.

try:
    conversations = load_conversations()

    conversation_id = st.query_params.get("conversation_id")

    existing_ids = {
        conversation["id"]
        for conversation in conversations
    }

    if not conversation_id or conversation_id not in existing_ids:
        conversation_id = create_conversation()

        st.query_params["conversation_id"] = conversation_id
        st.rerun()

except requests.RequestException as exc:
    st.error(f"Could not initialize conversation: {exc}")
    st.stop()


# Citation display

def show_sources(citations):
    if not citations:
        return

    with st.expander("Sources"):
        for citation in citations:
            st.markdown(
                f"**{citation.get('filename', 'PDF')} · "
                f"Page {citation.get('page_number', '?')}**"
            )

            excerpt = citation.get("excerpt", "")

            if excerpt:
                st.caption(excerpt[:1000])


def show_verification(verification):
    if not verification:
        return

    status = verification.get("status", "UNVERIFIED")

    if status == "VERIFIED":
        st.success("Citation check: Verified")

    elif status == "NEEDS_REVIEW":
        st.warning("Citation check: Needs review")

    elif status == "INSUFFICIENT_EVIDENCE":
        st.warning("Citation check: Insufficient evidence")

    elif status == "NOT_REQUIRED":
        st.caption("Citation check: Not required")

    elif status == "NO_CITATIONS":
        st.info("Citation check: No citations")

    else:
        st.info("Citation check: Unverified")


# Load current conversation data

try:
    messages = load_messages(conversation_id)
    documents = load_documents(conversation_id)

except requests.RequestException as exc:
    st.error(f"Could not load saved chat data: {exc}")
    st.stop()


# Sidebar

with st.sidebar:
    st.title("📄 PDF Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        try:
            new_id = create_conversation()

            st.query_params["conversation_id"] = new_id
            st.rerun()

        except requests.RequestException as exc:
            st.error(f"Could not create conversation: {exc}")

    st.divider()
    st.subheader("Previous conversations")

    for conversation in conversations:
        chat_id = conversation["id"]
        title = conversation["title"]

        if st.button(
            title,
            key=f"chat_{chat_id}",
            use_container_width=True,
            type=(
                "primary"
                if chat_id == conversation_id
                else "secondary"
            ),
        ):
            st.query_params["conversation_id"] = chat_id
            st.rerun()

    st.divider()
    st.subheader("PDFs in this chat")

    if documents:
        for document in documents:
            st.write(f"📄 {document['filename']}")
    else:
        st.caption("No PDF uploaded to this conversation.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key=f"pdf_{conversation_id}",
    )

    if st.button(
        "Upload PDF",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    ):
        try:
            with st.spinner("Uploading and processing PDF..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    data={
                        "conversation_id": conversation_id,
                    },
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=300,
                )

                response.raise_for_status()

            st.success(
                response.json().get(
                    "message",
                    "PDF uploaded successfully.",
                )
            )

            st.rerun()

        except requests.HTTPError:
            try:
                detail = response.json().get(
                    "detail",
                    "PDF upload failed.",
                )
            except ValueError:
                detail = response.text

            st.error(detail)

        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")


# Main chat

st.title("Ask your PDF")
st.caption("Upload a PDF and ask questions about its content.")
st.caption(f"Conversation ID: {conversation_id}")


# Display saved messages

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("citations") or [])

            show_verification(
                message.get("citation_verification") or {}
            )


# Question answering

question = st.chat_input("Ask a question about your PDF...")

if question:
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "query": question,
                    "conversation_id": conversation_id,
                },
                timeout=180,
            )

            response.raise_for_status()

        # Reload messages saved by the backend.
        st.rerun()

    except requests.HTTPError:
        try:
            detail = response.json().get(
                "detail",
                "Failed to generate an answer.",
            )
        except ValueError:
            detail = response.text

        st.error(detail)

    except requests.RequestException as exc:
        st.error(f"Backend connection failed: {exc}")
