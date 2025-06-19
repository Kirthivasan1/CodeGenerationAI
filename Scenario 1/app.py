
import streamlit as st
import time
from CodeModel import generate_code
st.header("CodeGenie: AI-powered code generator")

with st.form("my_form"):
    user_input = st.text_area("Enter your text prompt below and click the button to submit.")
    submit = st.form_submit_button(label="Submit text prompt")

if submit:
    with st.spinner(text="Generating code... It may take some time"):
        code, start, end = generate_code(prompt=user_input)
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        st.success(
            "Processing time: {:0>2}:{:0>2}:{:05.2f}.".format(
                int(hours), int(minutes), seconds
            )
        )
        st.code(code, language='python')

    st.sidebar.markdown("## Guide")
    st.sidebar.info("This tool uses salesforce codegen 350m parameters")

