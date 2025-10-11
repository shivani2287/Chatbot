import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

OPENAI_SECRET_KEY = "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"
#upload pdf file
st.header("My first chat bot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload your PDF and start asking questions", type="pdf")
    
text = ""
# Extract text from PDF
if file is not None:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    #st.write(text)  # Display first 500 characters of the extracted text


# Break text into chunks
text_splitter = RecursiveCharacterTextSplitter(
     separators="\n",
     chunk_size=1000,
     chunk_overlap=150,
     length_function=len
)
text_chunks = text_splitter.split_text(text=text)
#st.write(f"Total Chunks: {len(text_chunks)}")
#st.write(text_chunks)  # Display the first text chunk

#generate embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_SECRET_KEY)

#create vector store
vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
st.write(vectorstore)  # Display vector store information