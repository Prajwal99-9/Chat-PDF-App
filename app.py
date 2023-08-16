import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

#side  bar contents

with st.sidebar:
    st.title('Chat PDF App')
    st.markdown(''' 
                ## About
                This app is an Chat PDF App built using:
               -[Streamlit](https://streamlit.io/)
               -[Langchain](https://pyhton.langchain.com/)
               -[OpenAi](https://platform.openai.com)/docs/models)
            ''')
    add_vertical_space(5)
    st.write('Made with by prajwal kumbar' )
    
  
def main():
    st.header("Chat with pdf")
     
    
    #upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
   
    
    #st.write(pdf)
    if pdf is not None:
       pdf_reader = PdfReader(pdf)
       
       text = ""
       for page in pdf_reader.pages:
           text +=page.extract_text()
           
       
       text_splitter =RecursiveCharacterTextSplitter(
           chunk_size=1000,
           chunk_overlap=200,
           length_function=len
       )
       chunks= text_splitter.split_text(text=text)
       
       #embeddings
       
       store_name=pdf.name[:-4]
       
       if os.path.exists(f"{store_name}.pkl"):
           with open(f"{store_name}.pkl","rb") as f:
              VectorStore = pickle.load(f)
              #st.write('Embeddings Loaded from the disk')
       else:
           embeddings = OpenAIEmbeddings
           VectorStore = FAISS.from_text(chunks, embedding=embeddings,type='text')
           with open (f"{store_name}.pkl", "wb") as f: 
                pickle.dump(VectorStore,f)
           #st.write('Embeddings Computation completed')
       
       #Accept user question /query
       query=st.text_input("Ask question about your pdf sfile:")
       #st.write(query)run stre
    
       if query:
          docs = VectorStore.similarity_search(query=query)
          #st.write(docs)
          llm=OpenAI(model_name='gpt-3.5-turbo')
          chain=load_qa_chain(llm=llm, chain_type='stuff')
          with get_openai_callback() as cb:
               response=chain.run(input_documents=docs,question=query)
               print(cb)
          st.write(response)
       #st.write(chunks)
       
    
    
      #st.write(text) 



if __name__ == '__main__':
    main()