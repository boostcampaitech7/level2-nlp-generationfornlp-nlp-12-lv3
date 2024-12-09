from sentence_transformers import SentenceTransformer, util

def get_combined_docs_with_similarity(item, dense_retriever, bm25_retriever, mecab, k_bm25=20):
    """
    question과 각 choice별로 가장 유사도가 높은 문서들을 선택
    """
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    
    # BM25 검색
    query_tokens = bm25s.tokenize(' '.join(mecab.morphs(item['paragraph'])))
    bm25_results, bm25_scores = bm25_retriever.retrieve(query_tokens, k=k_bm25)
    bm25_docs = [result['text'] for result in bm25_results[0]]
    
    # 문서들을 임베딩
    doc_embeddings = model.encode(bm25_docs, convert_to_tensor=True)
    
    # Question 유사도 계산
    question_embedding = model.encode(item['question'], convert_to_tensor=True)
    question_similarities = util.pytorch_cos_sim(question_embedding, doc_embeddings)[0]
    
    # Choices 각각의 유사도 계산
    choices_embeddings = model.encode(item['choices'], convert_to_tensor=True)
    choices_similarities = util.pytorch_cos_sim(choices_embeddings, doc_embeddings)
    
    # 선택된 문서들을 저장할 set
    selected_docs = set()
    
    # Question에 대해 가장 유사도가 높은 문서 선택
    question_top_idx = torch.argmax(question_similarities).item()
    selected_docs.add(bm25_docs[question_top_idx])
    
    # print("\n=== Question 관련 최적 문서 ===")
    # print(f"질문: {item['question']}")
    # print(f"유사도: {float(question_similarities[question_top_idx]):.4f}")
    # print(f"선택된 문서: {bm25_docs[question_top_idx][:200]}...")
    
    # 각 Choice별로 가장 유사도가 높은 문서 선택
    # print("\n=== Choices 관련 최적 문서 ===")
    for idx, choice in enumerate(item['choices']):
        choice_similarities = choices_similarities[idx]
        top_idx = torch.argmax(choice_similarities).item()
        selected_docs.add(bm25_docs[top_idx])
        
        # print(f"\n선택지 {idx+1}: {choice}")
        # print(f"유사도: {float(choice_similarities[top_idx]):.4f}")
        # print(f"선택된 문서: {bm25_docs[top_idx][:200]}...")
    
    # 모든 선택된 문서 결합
    final_docs = list(selected_docs)
    
    # print(f"\n총 선택된 문서 수: {len(final_docs)}")
    
    # BM25 점수도 함께 표시
    # print("\n=== 선택된 문서들의 BM25 점수 ===")
    for doc in final_docs:
        doc_idx = bm25_docs.index(doc)
        # print(f"BM25 점수: {float(bm25_scores[0][doc_idx]):.4f}")
        # print(f"문서 내용: {doc[:200]}...\n")
    
    return "\n".join(final_docs)