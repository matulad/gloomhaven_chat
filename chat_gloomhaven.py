import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, render_template, request
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

_ = load_dotenv(find_dotenv())

ROOT_DIR = os.environ["ROOT_DIR"]
DOCUMENT_PATH = f"{ROOT_DIR}\\{os.environ['DOCUMENT_PATH']}"
PERSIST_DIR = f"{ROOT_DIR}\\gloomhaven_files\\faiss\\"

PARAGRAPH_CHAR = "\uf086"  # os.environ["PARAGRAPH_CHAR"]
SPLITTER_CHUNK_SIZE = 300
SPLITTER_CHUNK_OVERLAP = 0  # paragraphs are usually reasonably small


def _initialize_vectorstore(
    rulebook_path: str = DOCUMENT_PATH, persist_folder=PERSIST_DIR
):
    loader = PyMuPDFLoader(rulebook_path)
    all_docs = loader.load()
    docs = [
        doc for doc in all_docs if PARAGRAPH_CHAR in doc.page_content
    ]  # narrow down to glossary only

    splitter = RecursiveCharacterTextSplitter(
        separators=[PARAGRAPH_CHAR],
        keep_separator=False,
        chunk_size=SPLITTER_CHUNK_SIZE,
        chunk_overlap=SPLITTER_CHUNK_OVERLAP,
    )
    splitted_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(
        documents=splitted_docs,
        embedding=embeddings,
    )
    vectordb.save_local(persist_folder)

    return vectordb


def get_qa_chain():
    vectordb = _initialize_vectorstore(DOCUMENT_PATH, PERSIST_DIR)
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain


def _parse_source_docs(source_docs):
    pages = {}
    for doc in source_docs:
        file = doc.metadata["file_path"].split("/")[-1]
        page = str(doc.metadata["page"])
        if file in pages:
            pages[file] = ", ".join([pages[file], page])
        else:
            pages[file] = page

    source_info = ""
    for file, locations in pages.items():
        source_info += f"{file}, pages: {locations}\n"
    return source_info


qa_chain = get_qa_chain()


@app.route("/")
def index():
    return render_template("index.html", query="What would you like to know?")


@app.route("/search")
def search():
    query = request.args.get("query")
    result = qa_chain({"query": query})
    source_info = _parse_source_docs(result["source_documents"])
    return render_template(
        "search.html", query=query, answer=result["result"], source_info=source_info
    )


@app.route("/test")
def test():
    return render_template("test.html")


if __name__ == "__main__":
    app.run(debug=True)
