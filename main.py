import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from PyPDF2 import PdfReader
from langchain import HuggingFaceHub


def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PdfReader(file)
        # print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            pageObj.clear()
            text_list.append(text)
            sources_list.append(file.name + "_page_"+str(i))
    return [text_list, sources_list]


st.set_page_config(layout="centered", page_title="Multidoc_QnA")
st.header("Multidoc_QnA")
st.write("---")

# file uploader
uploaded_files = st.file_uploader(
    "Upload documents", accept_multiple_files=True, type=["txt", "pdf"])


if uploaded_files is None:
    st.info(f"""Upload files to analyse""")
elif uploaded_files:
    st.write(str(len(uploaded_files)) + " document(s) loaded...")

    textify_output = read_and_textify(uploaded_files)

    documents = textify_output[0]
    sources = textify_output[1]

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vStore = Chroma.from_texts(documents, embeddings, metadatas=[
                               {"source": s} for s in sources])

    model_name = "tiiuae/falcon-7b-instruct"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {"k": 2}

    # initiate model
    llm = HuggingFaceHub(
        huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"],
        repo_id=model_name,
        model_kwargs={
            # "task": "text2text-generation",
            "temperature": 0.8, "max_new_tokens": 100}

    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)

    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")

    if st.button("Get Response"):
        try:
            with st.spinner("model is working on it..."):
                result = chain({"question": user_q}, return_only_outputs=True)
                print(result)
                st.subheader("your response: ")
                st.write(result["answer"])
                st.subheader('Source pages:')
                st.write(result['sources'])
        except Exception as e:
            st.error(f"An error occured: {e}")
