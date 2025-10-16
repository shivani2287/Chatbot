import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.retrieval_qa import load_qa_chain

key_vault_name = "shivanikeys"
vault_url = f"https://{key_vault_name}.vault.azure.net/"

# # Authenticate and create a client
# credential = DefaultAzureCredential()
# client = SecretClient(vault_url=vault_url, credential=credential)

# # Fetch a secret
# secret_name = "your-secret-name"
# retrieved_secret = client.get_secret(secret_name)

#print(f"Secret Value: {retrieved_secret.value}")

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

    # get user query
    user_question = st.text_input("Ask questions about your PDF file:")

    # do similarity search
    if user_question:
        match = vectorstore.similarity_search(user_question)
        #st.write(docs)  # Display the retrieved documents
        
        #generate answer
        chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=OPENAI_SECRET_KEY), chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)  # Display the answer