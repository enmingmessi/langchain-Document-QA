import os
import apikey
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: dict

# os.environ['OPENAI_API_KEY'] = apikey.apikey
# embeddings = OpenAIEmbeddings(show_progress_bar=True)
model_name = "bge-large-zh"
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)

# separators = [r"\b\d+\.\d+\.\d+\b","(?<=\。)", ""]
separators = [r"\n(?=\d+\.\d+\.\d+)", ""]
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 2400,
    chunk_overlap  = 120,
    length_function = len,
    separators = separators
)

#将文档内容以Faiss向量数据库的方式进行indexing
for i, filename in enumerate(os.listdir('./knowledge/'),start=0):
# for i, filename in enumerate(os.listdir('./know/'), start=0):
    try:
        # loader = UnstructuredPDFLoader(f'./knowledge/{filename}',mode="single", strategy="auto")
        loader = PyPDFLoader(f'./knowledge/{filename}')
        # loader = PyPDFLoader(f'./know/{filename}')
        data = loader.load()
        for j in range(len(data)):
            data[j].page_content = data[j].page_content.replace(" ", "")
            data[j].page_content = data[j].page_content.replace("０", "0").replace("１", "1").replace("２", "2").replace("３", "3").replace("４", "4").replace(
            "５", "5").replace("６", "6").replace("７", "7").replace("８", "8").replace("９", "9").replace("．", ".")

        content = ''
        for d in data:
            content += d.page_content
        contents = re.split(r"\n(?=\d+\.\d+\.\d+)", content)

        metadata = {'source': f'./knowledge/{filename}'}
        results = []
        for j in contents:
            page_content = j
            result = Document(page_content=page_content, metadata=metadata)
            results.append(result)
        docs = text_splitter.split_documents(results)
        if i == 0:
            db = FAISS.from_documents(docs, embeddings)
        else:
            db1 = FAISS.from_documents(docs, embeddings)
            db.merge_from(db1)
    except Exception as e:
        print(f"Caught an exception: {e}")
        print(filename)

#将内容保存到本地向量数据库文件中
db.save_local("faiss_index_pyPDF_bge_split")

