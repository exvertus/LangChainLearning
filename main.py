from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

if not st.session_state.get("user_prompt_history", None):
    st.session_state["user_prompt_history"] = []

if not st.session_state.get("chat_answers_history", None):
    st.session_state["chat_answers_history"] = []

if not st.session_state.get("chat_history", None):
    st.session_state["chat_history"] = []

st.header("Langchain Document Helper")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])

        formatted_response = f"{generated_response['answer']}\n\n"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

if st.session_state["chat_answers_history"]:
    for chat_answer, user_prompt in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(chat_answer)
        message(user_prompt, is_user=True)