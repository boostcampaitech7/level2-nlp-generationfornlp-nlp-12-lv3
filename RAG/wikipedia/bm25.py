from konlpy.tag import Okt
from konlpy.tag import Mecab
import pandas as pd
from tqdm import tqdm
import bm25s

# tqdm과 pandas 결합
tqdm.pandas()

# Okt 객체 생성
okt = Okt()
# Mecab 객체 생성
mecab = Mecab()

# 데이터 로드
print("Loading preprocessed Wikipedia data...")
data = pd.read_csv('cleaned_wikipedia_paragraphs.csv')

# 전처리: 결측값 처리 및 빈 텍스트 제거
data['paragraph'] = data['paragraph'].fillna('')
data = data[data['paragraph'].str.strip() != '']

# 토큰화
print("Tokenizing documents with Mecab...")
data['tokens'] = data['paragraph'].progress_apply(lambda x: ' '.join(mecab.morphs(x)))
corpus = data['tokens'].tolist()

corpus_tokens = bm25s.tokenize(corpus)

# Create the BM25 model and index the corpus
print("Creating BM25 index...")
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# BM25 인덱스 저장
print("Saving BM25 index...")
retriever.save("wikipedia_index_bm25", corpus=corpus)
print("BM25 index saved as 'wikipedia_index_bm25'.")

# # 저장된 BM25 인덱스 로드
# print("Loading BM25 index and document data...")
# reloaded_retriever = bm25s.BM25.load("wikipedia_index_bm25", load_corpus=True)
# print("BM25 index and document data successfully loaded.")