from flask import Flask, request
from flask_cors import CORS
import os
import openai
from flask import jsonify

from PyPDF2 import PdfReader
from PyPDF2 import PdfMerger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from bs4 import BeautifulSoup



app = Flask(__name__)
CORS(app) 



OPENAI_API_KEY = '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'
base_conocimiento = None



@app.route('/cargar', methods=['POST'])
def upload_file():
    global base_conocimiento

    print("comenzando")

    # Verificar si se envio mas de un archivo PDF para unirlos en uno solo
    files = request.files.getlist('file')
    print(files)
    if len(files) > 1:
        print("leyendo")
        
        merger = PdfMerger()
        for pdf in files:
            merger.append(pdf)
     
        merged_filename = 'merged_file.pdf'
        merger.write(merged_filename)
        merger.close()
       
        base_conocimiento = create_embeddings(merged_filename)
        print('Archivos PDF unidos y procesados correctamente.')
        return jsonify("Archivos PDF unidos y procesados correctamente.")
    else:
        # Si llega solo un archivo PDF, procesarlo normalmente
        file = files[0]
        base_conocimiento = create_embeddings(file)
        print('Archivo PDF procesado correctamente.')
        return jsonify("Archivo PDF procesado correctamente.")
   



@app.route('/preguntapdf', methods=['POST'])
def pregunta():
    
    pregunta = request.form['texto']
        
    respuesta = envioGPT(base_conocimiento , pregunta)
    print(respuesta)

    return jsonify(respuesta)





def create_embeddingshtml(html_file):
    

    soup = BeautifulSoup(html_file.stream, 'html.parser')

   
    text = soup.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    base_conocimiento = FAISS.from_texts(chunks, embeddings)

    return base_conocimiento


def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    base_conocimiento = FAISS.from_texts(chunks, embeddings)

    return base_conocimiento



def envioGPT(base_conocimiento, pregunta):

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    paragraph = base_conocimiento.similarity_search(pregunta, 4)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    chain = load_qa_chain(llm, chain_type="stuff")
    respuesta = chain.run(input_documents=paragraph, question=pregunta)
    
    return respuesta


if __name__ == '__main__':
    # Cambia la dirección IP aquí
    app.run(debug=True, host='192.168.1.8', port=5000)



  
