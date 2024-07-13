import streamlit as st

st.title("Bạn cần hỏi về gì ?")
st.header("Hãy tải ảnh lên")
file  = st.file_uploader("",type=["png","jpg","jpeg"])
if file:
    st.image(file,use_column_width=True)

    #text input
user_question = st.text_input("Hay cho toi cau hoi cua ban")