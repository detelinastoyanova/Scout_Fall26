# app.py
# This file holds only the Streamlit interface. The agent logic lives in Scout_step2.py.
# pip install streamlit
# run with:  streamlit run app.py

import streamlit as st
from Scout_step5 import create_chat_history, ask_scout

company_logo = "images/the_keepsake.svg"
scout_icon = "images/scout_icon.svg"

AVATARS = {
    "user": "👀",
    "assistant": scout_icon
}


st.set_page_config(page_title="Scout - The Keepsake", page_icon=scout_icon)
st.image(company_logo)
st.caption("Business Analyst at The Keepsake")

# Streamlit reruns this whole script on every interaction, so the chat history
# is stored in st.session_state to keep the agent's memory between messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = create_chat_history()

# sidebar button to start a fresh conversation
with st.sidebar:
    if st.button("New conversation"):
        st.session_state.chat_history = create_chat_history()
        st.rerun()

# display the previous messages (skip the system prompt)
for message in st.session_state.chat_history:
    if message["role"] != "system":
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            st.markdown(message["content"])

# chat input box at the bottom of the page
user_prompt = st.chat_input("Ask Scout a question ...")

if user_prompt:
    # show the user's message right away
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_prompt)

    # get and show the agent's response (ask_scout also updates the memory)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Scout is thinking ..."):
            answer = ask_scout(st.session_state.chat_history, user_prompt)
        st.markdown(answer)