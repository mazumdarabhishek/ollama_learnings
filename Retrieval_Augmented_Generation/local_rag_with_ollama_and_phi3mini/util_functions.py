from langchain.chains.flare.prompts import PROMPT_TEMPLATE
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_ollama import OllamaLLM
import ollama
import os

from langchain_core.prompts import ChatPromptTemplate
from nltk.corpus.reader import documents

data_path = "data_sources"

def load_documents(data_path:str):
    loader = PyPDFDirectoryLoader(data_path, glob="*.pdf")
    documents = loader.load()
    return documents

def split_into_chunks(documents: list[Document]):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap= 200,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} were split into {len(chunks)}")

    sample = chunks[4]
    print(sample.page_content,'\n')
    print(sample.metadata)

    return chunks

def load_data_into_chroma(chunks, embedding_function, persist_dir, collection_name):

    db = Chroma.from_documents(
        documents= chunks,
        embedding= embedding_function,
        persist_directory= persist_dir,
        collection_name=collection_name)
    # db = Chroma(
    #     collection_name=collection_name,
    #     embedding_function=embedding_function,
    #     persist_directory=persist_dir)
    #
    # for ch in chunks:
    #     db.add_documents()
    db.persist()
    print("Data Loaded to Chroma Collection successfully")

def run_rag(model_name, embedding_model_name, chroma_persist_dir, collection_name, query):
    embedder = OllamaEmbeddings(model=embedding_model_name)
    vector_store = Chroma(
        collection_name= collection_name,
        embedding_function=embedder,
        persist_directory= chroma_persist_dir
    )
    query_embedding = embedder.embed_query(query)
    results = vector_store.similarity_search_by_vector_with_relevance_scores(
        query_embedding, k=3
    )
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    context = "\n\n---\n\n".join([doc.page_content for doc,_ in results])
    print("SYS: context retrived-----\n", context)
    PROPMT_TEMPLATE = """
    Answer the question based only on the following context:
    
    {context}
    
    ---
    
    Answer the question based on the above context: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(PROPMT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)

    model = OllamaLLM(model=model_name)

    result = model.invoke(prompt)
    print("AI: ", result)


def main():
    # documents = load_documents(data_path="data_sources")
    # chunks = split_into_chunks(documents=documents)
    # embedder = OllamaEmbeddings(model="mxbai-embed-large")
    # load_data_into_chroma(chunks=chunks,embedding_function=embedder,persist_dir="Chroma",
    #                       collection_name='Vegiterian_recepies')

    run_rag(model_name='phi3:latest', embedding_model_name="mxbai-embed-large",
            chroma_persist_dir="Chroma",collection_name='Vegiterian_recepies',
            query="Why is vegetarian food better for health?")

if __name__ == "__main__":
    main()
