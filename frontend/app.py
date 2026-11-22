
import requests
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


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


# Sidebar
with st.sidebar:
    st.title("📄 PDF Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
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

            result = response.json()
            st.success(
                result.get("message", "PDF uploaded successfully.")
            )

        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")


# Main chat
st.title("Ask your PDF")
st.caption("Upload a PDF and ask questions about its content.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("citations", []))
            show_verification(
                message.get("citation_verification", {})
            )


# Ask a question
question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": question},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()

            answer = result.get(
                "answer",
                "No answer returned by the backend.",
            )

            citations = result.get("citations", [])
            verification = result.get("citation_verification", {})

            st.markdown(answer)

            show_sources(citations)
            show_verification(verification)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "citation_verification": verification,
                }
            )

        except requests.HTTPError as exc:
            error_message = f"Request failed: {exc}"

            try:
                error_message = response.json().get(
                    "detail",
                    error_message,
                )
            except ValueError:
                pass

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

        except requests.RequestException as exc:
            error_message = f"Backend connection failed: {exc}"

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )
