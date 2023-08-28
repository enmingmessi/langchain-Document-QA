import os
import apikey
import langchain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.chains import StuffDocumentsChain, LLMChain
from examples import examples


langchain.debug = True

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ['OPENAI_API_KEY'] = apikey.apikey
# embeddings = OpenAIEmbeddings()
model_name = "bge-large-zh"
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)

load_db = FAISS.load_local("faiss_index_pyPDF_bge_split", embeddings)


lambda_mult_1 = 0.9
lambda_mult_2 = 0.55
retriever_1 = load_db.as_retriever(search_type='mmr', search_kwargs={"k": 20, 'lambda_mult': lambda_mult_1, 'fetch_k': 50})
retriever_2 = load_db.as_retriever(search_type='mmr', search_kwargs={"k": 20, 'lambda_mult': lambda_mult_2, 'fetch_k': 50})


#设定从向量数据库中查询数据的模式 & 使用的LLM模型
retriever = load_db.as_retriever(search_type='mmr',search_kwargs={"k": 20, 'lambda_mult': 0.9, 'fetch_k': 50})
# retriever = load_db.as_retriever(search_type='similarity',search_kwargs={"k": 20})
llm_quick = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
llm_smart = ChatOpenAI(model_name="gpt-4", temperature=0)

# 请尽可能使用背景信息中同样的语言描述方式，答案中要包含关键的数值、概念、法规等内容。\
# 给出答案的格式要清晰易读，使用合理的换行和编号。\
# 每个类似于'3.8.2'、'3.8.3'这样的标题之间内容为一段，一般找到正确内容可以之间返回对应标题内的整段内容，请注意，是整段内容。\
# 标题'3.8.2'中一般还会包含几个类似于'1'、'2'这样的小标题。\
# 假如背景信息中针对问题有明显矛盾的多个答案，请你同时给出所有答案内容并标注出每一个答案所来自的document的metadata出处。除了这种情况之外不需要给出来源。\
#自定义在Stuff回答模式下的提示词模板



# Use the background information below to answer the question posed. If you don't know the answer, answer that you don't know what the answer is. \
# Give the answer directly, without any opening phrases, and especially don't say anything like "Based on the context provided,..." \
# Please use the original text of the contextual information wherever possible, generally the answer will be contained in something like '3.8.2' after the heading. \
# If the answer is supplemented under the same heading in the background information, it is also returned together. \
# If there is more than one answer to the question in the background information, please give the answer that is closest to the keyword of the question. \
# If there is more than one answer to a question in the background information, please also give the metadata source of the document from which the answer comes. \
prompt_template = """使用下面的背景信息来回答所提出的问题。如果你不知道答案，就回答说你不知道答案是什么。 \
请直接给出答案，不要加任何开头语，尤其是不要说任何类似“根据提供的上下文,...”这样的内容。\
请尽可能使用背景信息的原文，一般来说，答案会包含在类似于'3.8.2'这样的标题后的内容中。\
如果背景信息中的同一标题下，对答案进行了补充，也一并返回。\
假如背景信息中针对问题有多个答案，请你给出和问题的关键词最接近的答案。\
请你同时给出答案所来自的document的metadata出处。\
Please think step by step。

背景信息：

{context}

所需回答的问题： {question}

Answer in Simplified Chinese :"""
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm_quick
)

retriever1_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever_1, llm=llm_quick
)
retriever2_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever_2, llm=llm_quick
)


def qachain(question):
    # docs = retriever_from_llm.get_relevant_documents(question)

    docs_1 = retriever1_from_llm.get_relevant_documents(question)
    docs_2 = retriever2_from_llm.get_relevant_documents(question)
    combined_docs = []
    for doc in docs_1:
        if doc not in combined_docs:
            combined_docs.append(doc)
    for doc in docs_2:
        if doc not in combined_docs:
            combined_docs.append(doc)
    docs = combined_docs

    # filtered_docs = [doc for doc in docs if '……………………' not in doc.page_content]
    print(docs)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    print(reordered_docs)
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    llm_chain = LLMChain(llm=llm_quick, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    result = chain.run(input_documents=reordered_docs, question=question)
    # result = chain.run(input_documents=docs, question=question)
    return result

#测试单个问题
# 装配式内装修工程应在什么时候进行室内环境质量验收工作？                  # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 装配式建筑评价等级如何划分？                                       # "k": 20, 'lambda_mult': 0.55, 'fetch_k': 50   1
# 装配式住宅设计选型应包括哪些内容？0.55       设计选型应包括哪些内容？   # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 桁架钢筋混凝土叠合板应满足什么要求？                                # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 装配整体式剪力墙结构应如何布置？ 1                                 # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 无障碍通道的宽度应如何确定？                                      # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 博物馆建筑在进行总平面设计时，应符合哪些规定？                       # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 在设计物流建筑时，场地设计标高应符合哪些规定？                       # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 公园的停车场应如何布置？ 1                                       # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
# 剧场设计中，主舞台的天桥应如何设计？                               # "k": 20, 'lambda_mult': 0.9, 'fetch_k': 50   1
question = "装配式建筑评价等级如何划分？"
print(qachain(question))


'''
# 用GPT-4来进行综合的对比测试评判
eval_chain = QAEvalChain.from_llm(llm_quick)
predictions = []
for i in examples:
    result = qachain(question=i['query'])
    result_dict = {'query':i['query'],'answer':i['answer'],'result':result}
    predictions.append(result_dict)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    # print("Predicted Grade: " + graded_outputs[i]['results'])
    print()
'''