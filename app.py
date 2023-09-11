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



def main():
    st.header("Chat with PDF")
    load_dotenv()
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

        if query:
            docs = vectorStore.similarity_search(query, k=3)

            llm = OpenAI(model_name = 'gpt-3.5-turbo',temperature=0)
            chain = load_qa_chain(llm=llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question=query)
                print(cb)
            st.write(response)
            # query = st.text_input("Type your question:")
            # continue
        
        

    else:
        st.write("No PDF uploaded")

if __name__ =='__main__':
    main()