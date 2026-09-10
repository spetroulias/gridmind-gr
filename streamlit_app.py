import streamlit as st
import requests
import json

# Ρύθμιση σελίδας
st.set_page_config(
    page_title="GridMind AI Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Backend URL Configuration
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.title("⚡ GridMind Energy AI Dashboard")

# Sidebar Navigation για όλα τα Endpoints
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Επιλογή Λειτουργίας:",
    ["💬 Hybrid Chatbot (RAG)", "📊 Direct Load Data Query", "📄 Upload & Index Documents", "⚙️ System Health"]
)

# ---------------------------------------------------------
# PAGE 1: Hybrid Chatbot (RAG)
# ---------------------------------------------------------
if page == "💬 Hybrid Chatbot (RAG)":
    st.header("💬 GridMind RAG Assistant")
    st.caption("Ρώτησε σε φυσική γλώσσα για δεδομένα κατανάλωσης ή έγγραφα του συστήματος.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📌 Πηγές Δεδομένων"):
                    for src in message["sources"]:
                        st.write(f"- `{src}`")

    # React to user input
    if prompt := st.chat_input("π.χ. Ποια ήταν η κατανάλωση στις 3 Φεβρουαρίου 2026;"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call FastAPI Endpoint
        with st.chat_message("assistant"):
            with st.spinner("Ανάκτηση δεδομένων & παραγωγή απάντησης..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={"message": prompt},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "Δεν παραλήφθηκε απάντηση.")
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander("📌 Πηγές Δεδομένων"):
                                for src in sources:
                                    st.write(f"- `{src}`")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"Σφάλμα API: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Αποτυχία σύνδεσης με το Backend: {e}")

# ---------------------------------------------------------
# PAGE 2: Direct Load Data Query
# ---------------------------------------------------------
elif page == "📊 Direct Load Data Query":
    st.header("📊 Απευθείας Αναζήτηση στη Βάση (PostgreSQL)")
    
    selected_date = st.date_input("Επιλογή Ημερομηνίας:")
    
    if st.button("🔍 Ανάκτηση Φορτίου", type="primary"):
        date_str = selected_date.strftime("%Y-%m-%d")
        with st.spinner(f"Ανάκτηση δεδομένων για {date_str}..."):
            try:
                # Αν έχεις dedicated GET endpoint για ημερομηνία
                response = requests.get(f"{API_BASE_URL}/data/load", params={"date": date_str})
                
                if response.status_code == 200:
                    res_data = response.json()
                    st.success("Τα δεδομένα ανακτήθηκαν επιτυχώς!")
                    st.json(res_data)
                else:
                    st.warning(f"Δεν βρέθηκαν δεδομένα ή το endpoint επέστρεψε: {response.status_code}")
            except Exception as e:
                st.error(f"Σφάλμα σύνδεσης: {e}")

# ---------------------------------------------------------
# PAGE 3: Upload & Index Documents
# ---------------------------------------------------------
elif page == "📄 Upload & Index Documents":
    st.header("📄 Εισαγωγή Εγγράφων στο Qdrant Vector Store")
    
    uploaded_file = st.file_uploader("Επιλέξτε αρχείο (PDF, TXT):", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("🚀 Upload & Vectorize", type="primary"):
            with st.spinner("Επεξεργασία & δημιουργία Embeddings..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Το αρχείο '{uploaded_file.name}' ευρετηριάστηκε επιτυχώς στο Qdrant!")
                        st.json(response.json())
                    else:
                        st.error(f"Σφάλμα κατά το upload: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Αποτυχία σύνδεσης: {e}")

# ---------------------------------------------------------
# PAGE 4: System Health
# ---------------------------------------------------------
elif page == "⚙️ System Health":
    st.header("⚙️ Κατάσταση Συστήματος & Services")
    
    if st.button("🔄 Έλεγχος Healthcheck"):
        try:
            response = requests.get("http://127.0.0.1:8000/health")
            if response.status_code == 200:
                st.success("Όλες οι υπηρεσίες (PostgreSQL, Qdrant, Groq LLM) είναι ONLINE!")
                st.json(response.json())
            else:
                st.error(f"Healthcheck Failed: {response.status_code}")
        except Exception as e:
            st.error(f"Backend Server Offline: {e}")