import streamlit as st
from NLP import BilingualTranslationPipeline
import time


st.set_page_config(
    page_title="Arabic-English Machine Translation System",
    page_icon="🌐",
    layout="wide"
)



@st.cache_resource
def load_pipeline():
    pipeline = BilingualTranslationPipeline('ar-en')
    pipeline.setup_resources()
    pipeline.load_data()
    pipeline.load_model()
    return pipeline


# gui
def main():

    pipeline = load_pipeline()


    st.title("🌐 Arabic-English Translation System")


    tab1, tab2 = st.tabs(["Translation", "About app"])

    with tab1:

        st.header("Eter text for translation")
        input_text = st.text_area(
            "Arabic text",
            height=150,
            placeholder="Write arabic text here ....",
            label_visibility="collapsed"
        )


        col1, col2 = st.columns(2)
        with col1:
            translate_btn = st.button("Translate", use_container_width=True)
        with col2:
            clear_btn = st.button("Delete", use_container_width=True)

        if clear_btn:
            input_text = ""
            st.experimental_rerun()

        if translate_btn and not input_text:
            st.warning("Please enter the text for translation")


        if translate_btn and input_text:
            with st.spinner("Translation in progress"):
                start_time = time.time()
                result = pipeline.translate_text(input_text)
                elapsed_time = time.time() - start_time

            st.success("Translation successful!")


            result_tab1, result_tab2, result_tab3 = st.tabs([
                "Original Text",
                "Translated Text",
                "Translation Details"
            ])

            with result_tab1:
                st.subheader("Original Text")
                st.text_area(
                    "Original Text",
                    value=result['original'],
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )

            with result_tab2:
                st.subheader("Translated Text")
                st.text_area(
                    "Translated Text",
                    value=result['translated'],
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )

            with result_tab3:
                st.subheader("Translation details")
                st.write(f"**Text after cleaning :** {result['cleaned']}")
                st.write(f"**Translation time:** {elapsed_time:.2f} second")

    with tab2:
        st.header("About app")
        st.write("""
        ### Arabic-English Machine Translation System

        This application uses the Helsinki-NLP/opus-mt-ar-en model for machine translation from Arabic to English.

        **Features:**
        - Translation of Arabic texts to English
        - Preprocessing of text to improve translation quality
        - Easy-to-use user interface

        **How to use:**
        1. Write the Arabic text in the text box
        2. Click the "Translate" button
        3. View the results in the different sections

        """)


if __name__ == "__main__":
    main()