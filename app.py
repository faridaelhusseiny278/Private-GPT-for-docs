import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub,GPT4All, LlamaCpp, OpenAI
import os
import argparse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
#take pdf docs (list of pdf files) and return text with all the contents of the pdfs
def get_pdf_text(pdf_docs):
    text = "" #stores all of the text of the pdfs
    for pdf in pdf_docs:
        #creates pdf object that has pages 
        pdf_reader = PdfReader(pdf)
        #looping through the pages 
        for page in pdf_reader.pages:
            #extracts raw text from pdf 
            text += page.extract_text()
    return text
def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

def get_text_chunks(raw_text):
    #returns a list of chunks each is about 1000 characters long
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size= 1000, # 1000 characters
        chunk_overlap= 200, #to predict the last word of the previous chunk (in order to not lose the meaning of the sentence if it stopped halfway through)
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
def get_vectorstore(text_chunks):
    #returns a vector store of the text chunks 

    # embeddings = OpenAIEmbeddings()

    embeddings = HuggingFaceInstructEmbeddings(
        model_name = "hkunlp/instructor-xl" )
    vector_store = FAISS.from_texts(texts= text_chunks, embedding= embeddings)
    return vector_store
def get_conversation_chain(vector_store):
    # llm = ChatOpenAI()
    args = parse_arguments()

    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    # llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature" : 0.5,"max_length":512})    
    memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm( 
    llm = llm,
    retriever= vector_store.as_retriever(),
    memory= memory,
    )
    return conversation_chain
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i , message in enumerate (st.session_state.chat_history):
        if i%2 ==0 :
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def main():
    print("hi")
    load_dotenv()
    #set page config 
    st.set_page_config(page_title='Private-GPT', page_icon='🧊', layout='wide', initial_sidebar_state='auto')
    
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Private-GPT")
    user_question= st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs= st.file_uploader("Upload your documents", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
            #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
            #get the text chunks 
                text_chunks = get_text_chunks(raw_text)
            # create vector store (embeddings)
                vector_store = get_vectorstore(text_chunks)
            #  create conversation chain 
            # in order to save the conversation chain for the user
                st.session_state.conversation = get_conversation_chain(vector_store)

        

if __name__=='__main__':
    main()
