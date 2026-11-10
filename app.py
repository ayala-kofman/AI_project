"""
Streamlit UI for the organizational knowledge bot.
Run with:  streamlit run app.py
"""


import streamlit as st
from pathlib import Path
from rag_engine import build_index, load_index, answer_question, index_is_empty

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="בוט ידע ארגוני",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 בוט ידע ארגוני")
st.caption("שואל שאלות ועונה אך ורק מתוך המסמכים הארגוניים")

# ── Sidebar: index management ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ הגדרות")

    if st.button("📂 אנדקס מחדש את המסמכים", use_container_width=True):
        with st.spinner("מעבד מסמכים ובונה אינדקס..."):
            try:
                st.session_state["index"] = build_index()
                st.success("האינדקס נבנה בהצלחה!")
            except FileNotFoundError as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("**מסמכים בתיקיית data/**")
    docs = list(DATA_DIR.iterdir()) if DATA_DIR.exists() else []
    if docs:
        for d in docs:
            st.markdown(f"- {d.name}")
    else:
        st.info("אין מסמכים. העלה קבצים לתיקיית data/")

    st.markdown("---")
    st.markdown("**הגדרות RAG:**")
    st.markdown("• Chunk size: **512 טוקנים**")
    st.markdown("• Overlap: **50 טוקנים**")
    st.markdown("• TOP K: **3 וקטורים**")
    st.markdown("• Query expansion: **פעיל**")
    st.markdown("• מודל: **gemini-1.5-flash**")

# ── Load or init index ──────────────────────────────────────────────────────
if "index" not in st.session_state:
    if not index_is_empty():
        with st.spinner("טוען אינדקס קיים..."):
            st.session_state["index"] = load_index()
    else:
        st.info(
            "👈 לא נמצא אינדקס. הוסף מסמכים לתיקיית **data/** ולחץ **אנדקס מחדש** בסרגל הצד."
        )

# ── Chat history ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ──────────────────────────────────────────────────────────────
user_input = st.chat_input("שאל שאלה על המסמכים הארגוניים...")

if user_input:
    if "index" not in st.session_state:
        st.warning("בנה אינדקס תחילה (לחץ על הכפתור בסרגל הצד)")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("מחפש במסמכים..."):
                result = answer_question(st.session_state["index"], user_input)

            answer = result["answer"]
            st.markdown(answer)

        full_response = answer
        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )
