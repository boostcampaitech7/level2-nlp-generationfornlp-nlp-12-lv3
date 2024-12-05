from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import faiss
import pickle

class DenseRetriever:
    def __init__(self, model_name='jhgan/ko-sbert-nli'):
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.corpus = None
        
    def index(self, corpus):
        print("Encoding corpus with SBERT...")
        # 문서를 dense vector로 변환
        self.corpus = corpus
        embeddings = self.model.encode(
            corpus,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        # FAISS 인덱스 생성
        print("Building FAISS index...")
        embeddings = embeddings.cpu().numpy()
        dimension = embeddings.shape[1]
        
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
    def retrieve(self, query, k=5):
        # 쿼리를 dense vector로 변환
        query_vector = self.model.encode(query)
        
        # FAISS로 유사도 검색 - k를 더 크게 잡아서 중복 제거 후에도 충분한 결과가 있도록 함
        distances, indices = self.faiss_index.search(
            query_vector.reshape(1, -1).astype('float32'), 
            k * 2  # 중복 제거를 위해 더 많은 결과를 가져옴
        )
        
        results = []
        scores = []
        seen_texts = set()  # 중복 체크를 위한 집합
        
        # 중복을 제거하면서 결과 추가
        for idx, distance in zip(indices[0], distances[0]):
            text = self.corpus[idx]
            if text not in seen_texts:  # 중복되지 않은 텍스트만 추가
                seen_texts.add(text)
                results.append({'text': text})
                scores.append(1 / (1 + distance))  # 거리를 유사도 점수로 변환
                
                if len(results) == k:  # 원하는 개수만큼 찾았으면 중단
                    break
        
        return [results], scores
    
    def save(self, path, corpus=None):
        # FAISS 인덱스 저장
        faiss.write_index(self.faiss_index, f"{path}.faiss")
        
        # 코퍼스 저장
        if corpus is not None:
            with open(f"{path}.corpus", 'wb') as f:
                pickle.dump(corpus, f)
    
    @classmethod
    def load(cls, path, load_corpus=True):
        retriever = cls()
        
        # FAISS 인덱스 로드
        retriever.faiss_index = faiss.read_index(f"{path}.faiss")
        
        # 코퍼스 로드
        if load_corpus:
            with open(f"{path}.corpus", 'rb') as f:
                retriever.corpus = pickle.load(f)
                
        return retriever

# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    print("Loading preprocessed Wikipedia data...")
    data = pd.read_csv('cleaned_wikipedia_paragraphs.csv')
    # data = data.head(1000)
    
    # 전처리: 결측값 처리 및 빈 텍스트 제거
    data['paragraph'] = data['paragraph'].fillna('')
    data = data[data['paragraph'].str.strip() != '']
    corpus = data['paragraph'].tolist()
    
    # Dense Retriever 생성 및 인덱싱
    print("Creating Dense Retriever...")
    retriever = DenseRetriever()
    retriever.index(corpus)
    
    # 인덱스 저장
    print("Saving Dense index...")
    retriever.save("wikipedia_index_dense", corpus=corpus)
    print("Dense index saved as 'wikipedia_index_dense'.")