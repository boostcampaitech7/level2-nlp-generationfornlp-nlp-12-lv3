import bm25s
from dense_retriever import DenseRetriever
from konlpy.tag import Mecab
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_retrievers():
    print("Loading BM25 retriever...")
    bm25_retriever = bm25s.BM25.load("wikipedia_index_bm25", load_corpus=True)
    
    print("Loading Dense retriever...")
    dense_retriever = DenseRetriever.load("wikipedia_index_dense", load_corpus=True)
    
    print("Loading SentenceTransformer...")
    st_model = SentenceTransformer('jhgan/ko-sbert-nli')
    
    return bm25_retriever, dense_retriever, st_model

def get_hybrid_results(query, question, choices, bm25_retriever, st_model, mecab, k=5):
    """
    query, question과 choices를 모두 고려하여 가장 관련성 높은 문서들을 반환
    """
    # BM25로 후보 문서 검색
    query_tokens = bm25s.tokenize(' '.join(mecab.morphs(query)))
    bm25_results, bm25_scores = bm25_retriever.retrieve(query_tokens, k=k*4)
    bm25_docs = [result['text'] for result in bm25_results[0]]
    
    # 문서들을 임베딩
    doc_embeddings = st_model.encode(bm25_docs, convert_to_tensor=True)
    
    # Paragraph 유사도 계산
    paragraph_embedding = st_model.encode(query, convert_to_tensor=True)
    paragraph_similarities = util.pytorch_cos_sim(paragraph_embedding, doc_embeddings)[0]
    
    # Question 유사도 계산
    question_embedding = st_model.encode(question, convert_to_tensor=True)
    question_similarities = util.pytorch_cos_sim(question_embedding, doc_embeddings)[0]
    
    # Choices 각각의 유사도 계산
    choices_embeddings = st_model.encode(choices, convert_to_tensor=True)
    choices_similarities = util.pytorch_cos_sim(choices_embeddings, doc_embeddings)
    
    # 선택된 문서들을 저장할 set
    selected_docs = set()
    selected_scores = {}
    
    # Paragraph에 대해 가장 유사도가 높은 문서 선택
    paragraph_top_idx = torch.argmax(paragraph_similarities).item()
    selected_docs.add(bm25_docs[paragraph_top_idx])
    selected_scores[bm25_docs[paragraph_top_idx]] = float(paragraph_similarities[paragraph_top_idx])
    
    # Question에 대해 가장 유사도가 높은 문서 선택
    question_top_idx = torch.argmax(question_similarities).item()
    selected_docs.add(bm25_docs[question_top_idx])
    selected_scores[bm25_docs[question_top_idx]] = max(
        selected_scores.get(bm25_docs[question_top_idx], 0),
        float(question_similarities[question_top_idx])
    )
    
    # 각 Choice별로 가장 유사도가 높은 문서 선택
    for idx, choice in enumerate(choices):
        choice_similarities = choices_similarities[idx]
        top_idx = torch.argmax(choice_similarities).item()
        selected_docs.add(bm25_docs[top_idx])
        selected_scores[bm25_docs[top_idx]] = max(
            selected_scores.get(bm25_docs[top_idx], 0),
            float(choice_similarities[top_idx])
        )
    
    final_docs = list(selected_docs)
    final_scores = [selected_scores[doc] for doc in final_docs]
    
    return final_docs, final_scores

def compare_retrievers(query, question, choices, bm25_retriever, dense_retriever, st_model, k=5):
    mecab = Mecab()
    
    # BM25 검색
    query_tokens = bm25s.tokenize(' '.join(mecab.morphs(query)))
    bm25_results, bm25_scores = bm25_retriever.retrieve(query_tokens, k=k)
    
    # Dense 검색
    dense_results, dense_scores = dense_retriever.retrieve(query, k=k)
    
    print("=== Query ===")
    print(query)
    
    print("\n=== BM25 Results ===")
    for i in range(len(bm25_results[0])):
        print(f"Score: {float(bm25_scores[0][i]):.4f}")
        print(f"Text: {bm25_results[0][i]['text'][:200]}...")
        print()
        
    print("\n=== Dense Results ===")
    for i in range(len(dense_results[0])):
        print(f"Score: {float(dense_scores[i]):.4f}")
        print(f"Text: {dense_results[0][i]['text'][:200]}...")
        print()
    
    print("\n=== Hybrid Results ===")
    hybrid_docs, hybrid_scores = get_hybrid_results(
        query, question, choices, bm25_retriever, st_model, mecab, k=k
    )
    for doc, score in zip(hybrid_docs, hybrid_scores):
        print(f"Semantic Similarity Score: {score:.4f}")
        print(f"Text: {doc[:200]}...")
        print()

if __name__ == "__main__":
    # Retriever 로드
    bm25_retriever, dense_retriever, st_model = load_retrievers()
    
    # 테스트를 위한 예시 데이터
    query = "○장수왕은 남 진 정책의 일환으로 수도를 이곳으로 천도 하였다. ○묘청은 이곳으로 수도를 옮길 것을 주장하였다."
    question = "밑줄 친 ‘이곳’에 대한 설명으로 옳은 것은?"
    choices = ['쌍성총관부가 설치되었다 .', '망이 ㆍ망소이가 반란을 일으켰다 .', '제너럴 셔먼호 사건이 발생하였다 .', '1923년 조선 형평사가 결성되었다 .']
    
    compare_retrievers(query, question, choices, bm25_retriever, dense_retriever, st_model)
    print("\n" + "="*80 + "\n")