import streamlit  as st
import folderQA_reorder


if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

st.set_page_config(page_title="Welcome to 小建助手", layout="wide")

st.title("Welcome to 小建助手")

if "messages_human" not in st.session_state:
    st.session_state["messages_human"] = []
if "messages_assistant" not in st.session_state:
    st.session_state["messages_assistant"] = []

if st.session_state["OPENAI_API_KEY"]:
    with st.container():
        st.header("文档问答")

        for message in st.session_state["messages_human"]:
            with st.chat_message("user"):
                st.markdown(message)
        for message in st.session_state["messages_assistant"]:
            with st.chat_message("assistant"):
                st.markdown(message)

        question = st.chat_input("请输入问题，小建将为你解答")
        if question:
            st.session_state["messages_human"].append(question)
            with st.chat_message("user"):
                st.markdown(question)
            answer = folderQA_reorder.qachain(question)
            st.session_state["messages_assistant"].append(answer)
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    with st.container():
        st.warning("请先输入OpenAI API Key")