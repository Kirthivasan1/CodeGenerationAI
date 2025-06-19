
import streamlit as st
import time
from CodeModel import generate_code
st.header("CodeGenie: AI-powered code assistant")

with st.form("my_form"):
    user_input = st.text_area("Enter your text prompt below and click the button to submit.")
    submit = st.form_submit_button(label="Submit text prompt")

if submit:
    with st.spinner(text="Generating code... It may take some time"):
        code = generate_code(user=user_input)
        print(code)
        st.code(code, language='python')
    st.sidebar.markdown("## Guide")
    st.sidebar.info("This tool uses CodeLlama 7B parameters")