import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.image import UnstructuredImageLoader
import pickle
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback
import json

import pypdfium2 as pdfium
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from pytesseract import image_to_string

# side bar
with st.sidebar:
    st.title('PDF chat App')
    st.markdown("""
    # Capabilities of App
    Reads PDF and 
    Answers Question.
    """)
    add_vertical_space(5)
    st.write("""It's not who I am .. But what I do that defines me. --Batman""")

def convert_pdf_to_images(pdf_file, scale=300/72):

    pdf = pdfium.PdfDocument(pdf_file)
    page_indices = [i for i in range(len(pdf))]

    renderer = pdf.render(
        pdfium.PdfBitmap.to_pil,
        page_indices = page_indices,
        scale = scale
    )

    final_images =[]

    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i:image_byte_array}))

    return final_images

def extract_text_from_image(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    text = ""
    
    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        text = text+raw_text
    
    return text

def split_text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function=len
            )
    chunks = text_splitter.split_text(text=text)
    return chunks
    
def create_vectors(chunks):
    pdf_name = 'All_Hands_2023'
    if os.path.exists(f"{pdf_name}.pkl"):
        with open(f"{pdf_name}.pkl", 'rb') as f:
            vectorStore = pickle.load(f)
        st.write("Embedding already available for pdf.")
        return vectorStore
    else:
        #embeddings object
        st.write("Using OpenAi to create embeddings.")
        embeddings = OpenAIEmbeddings()
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{pdf_name}.pkl","wb") as f:
            pickle.dump(vectorStore,f)
        return vectorStore

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

def main():

    st.header("Chat with PDF")
    load_dotenv()
    file_path = "history.txt"
    convert_pdf_to_img = convert_pdf_to_images('All_Hands_2023.pdf')
    final_text = extract_text_from_image(convert_pdf_to_img)
    text_chunks = split_text_to_chunks(final_text)
    vec = create_vectors(text_chunks)
        
    # question box
    st.write("Howdy Partner!..What do you want to know about that PDF you uploaded:")
    query = st.text_input("Type your question:")
    #st.write(query)

    chat_history=[]
    if query:
        print('in while')
        docs = vec.similarity_search(query, k=3)

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
    print('running main')
    main()
    