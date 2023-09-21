import streamlit as st
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

def clear_file_content(file_path):
    with open(file_path, "w") as file:
        file.truncate(0)

def main():
    st.header("Chat with PDF")
    load_dotenv()
    file_path = "history.txt"
    
    #upload PDF
    pdf = st.file_uploader("Upload your PDF",type='pdf')
    
    #read pdf only when available
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name[:-4]
        #st.write(pdf_reader)
        text = ""
        
        for page in pdf_reader.pages:
            text = text+page.extract_text()
        #st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        #st.write(chunks)
        
        if os.path.exists(f"{pdf_name}.pkl"):
            with open(f"{pdf_name}.pkl", 'rb') as f:
                vectorStore = pickle.load(f)
            st.write("Embedding already available for pdf.")
        else:
            #embeddings object
            st.write("Using OpenAi to create embeddings.")
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{pdf_name}.pkl","wb") as f:
                pickle.dump(vectorStore,f)

        # question box
        st.write("Howdy Partner!..What do you want to know about that PDF you uploaded:")
        query = st.text_input("Type your question:")
        #st.write(query)

        chat_history=[]
        if query:
            print('in while')
            docs = vectorStore.similarity_search(query, k=3)

            llm = OpenAI(model_name = 'gpt-4',temperature=0)
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
                       
    else:
        st.write("No PDF uploaded")
    

if __name__ =='__main__':
    # print('creating ch')
    # chat_history=[]
    print('running main')
    main()
    #print(chat_history)