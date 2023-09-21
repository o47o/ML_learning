import streamlit as st
import PyPDF2 as pp 
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback
import json
import glob

# side bar
with st.sidebar:
    st.title('PDF chat App')
    st.markdown("""
    # Capabilities of App
    Reads PDF and 
    Answers Question.
    """)
    add_vertical_space(5)
    st.write('Madness is like Gravity.. all you need is a push.')


def chat_hist(file_path,chat_dict_list):
    if os.path.exists(file_path):
    # check size
        file_size = os.path.getsize(file_path)
        if file_size ==0:
            print('history file is empty')
            hist = chat_dict_list
            with open(file_path, "w") as file:
                json.dump(hist, file)
        else:
            print('history already present')
            with open(file_path, "r") as file:
                loaded_list_of_dicts = json.load(file)
            new_chat = chat_dict_list
            new_hist = loaded_list_of_dicts + new_chat
            with open(file_path, "w") as file:
                json.dump(new_hist, file) 

def pocessing_hist(processed_pdf_path, processed_pdf_list):
    if os.path.exists(processed_pdf_path):
    # check size
        file_size = os.path.getsize(processed_pdf_path)
        if file_size ==0:
            print('processing pdf history file is empty')
            hist = processed_pdf_list
            with open(processed_pdf_path, "w") as file:
                json.dump(hist, file)
        else:
            print('processing pdf history already present')
            with open(processed_pdf_path, "r") as file:
                loaded_list_of_pdfs = json.load(file)
            new_pdf_list = processed_pdf_list
            new_pdf_list = loaded_list_of_pdfs + new_pdf_list
            with open(processed_pdf_path, "w") as file:
                json.dump(new_pdf_list, file)

def clear_file_content(file_path):
    with open(file_path, "w") as file:
        file.truncate(0)

def main():
    st.header("Chat with PDF")
    load_dotenv()
    file_path = "history.txt"
    processed_pdf_path = "processed_pdf.txt"
    
    #upload PDF
    # pdf = st.file_uploader("Upload your PDF",type='pdf')
    
    pdf_dir = '/home/shashank/ml_app'
    
    pdf_pattern = os.path.join(pdf_dir,'*.pdf')

    pdf_files = glob.glob(pdf_pattern)

    pdf_processed=[]
    pdf_content =[]
    loaded_list_of_pdfs=[]

    text = ""

    # checking previously processed PDFs

    processing_file_size = os.path.getsize(processed_pdf_path)
    if processing_file_size >0:
        with open(processed_pdf_path, "r") as file:
            loaded_list_of_pdfs = json.load(file)
            loading_hist = True

    for file in pdf_files:
        #st.write("processing :",file)
        pdf_name = file[len(pdf_dir)+1:-4]
        #st.write(pdf_name)
        pdf_processed.append(pdf_name)
        pdf_path = file
        
        if pdf_name not in loaded_list_of_pdfs:
            st.write("Processing new PDFs")
            with open(pdf_path,'rb') as pdf:
                pdf_reader = pp.PdfReader(pdf)

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    pdf_content.append(page_text)
                    # text = text+page.extract_text()
        else:
            st.write("No files to process")
    # st.write(len(pdf_content))
    text =''.join(pdf_content)
    # st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function=len
            )
    chunks = text_splitter.split_text(text=text)

    pdf_vec_file_name = '_'.join(pdf_processed)

    if os.path.exists(f"{pdf_vec_file_name}.pkl"):
        with open(f"{pdf_vec_file_name}.pkl", 'rb') as f:
            vectorStore = pickle.load(f)
        st.write("Embedding already available for pdf.")
    else:
        st.write("Using OpenAi to create embeddings.")
        embeddings = OpenAIEmbeddings()
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{pdf_vec_file_name}.pkl","wb") as f:
            pickle.dump(vectorStore,f)

    # saving list of pdfs processed
    processing_file_size = os.path.getsize(processed_pdf_path)
    if processing_file_size >0:
        with open(processed_pdf_path, "r") as file:
            loaded_list_of_pdfs = json.load(file)
        if pdf_processed != loaded_list_of_pdfs:
            st.write("new pdfs")
            pocessing_hist(processed_pdf_path,pdf_processed)
        else: 
            st.write("No new PDFs")
    else:
        st.write("Processing history file is empty")
        pocessing_hist(processed_pdf_path,pdf_processed)

    # question box
    context = ';'.join(pdf_processed)
    st.write("Context is:")
    st.write(context)
    st.write("Howdy Partner!..What do you want to know about that PDFs mentioned above.")
    query = st.text_input("Type your question:")
    
    chat_history=[]
    if query:
        print('in while')
        docs = vectorStore.similarity_search(query, k=3)

        llm = OpenAI(model_name = 'gpt-3.5-turbo',temperature=0)
        chain = load_qa_chain(llm=llm, chain_type = "stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question=query)
            print(cb)
        st.write(response)
        # query = st.text_input("Type your question:")
        # continue
        chat_history.append({"You:":query,"LLM:":response})
        chat_hist(file_path,chat_history)
        # print(chat_history)

        st.subheader("Chat History")
        with open(file_path,'r') as file:
            loaded_hist = json.load(file)
        for item in loaded_hist:
            st.write(f"You: {item['You:']}")
            st.write(f"LLM: {item['LLM:']}")
   

if __name__ =='__main__':
    # print('creating ch')
    # chat_history=[]
    print('running main')
    main()
    #print(chat_history)