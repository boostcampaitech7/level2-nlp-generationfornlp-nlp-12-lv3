# 수능형 문제 풀이 모델

## 프로젝트 개요
- 목표 : LLM을 활용하여 AI 모델이 어떻게 하면 얼마나 다양한 분야의 시험에서 우수한 성적을 받을 수 있을 지에 대해 연구해보고, 한국어와 수능형 문제에 특화된 모델을 개발하여 소규모 모델로 대형모델 수준의 성능을 달성하는 것.
- 평가 기준 : Accuracy<span style="color:gray">  = 모델이 맞춘 수 / 전체 문제수</span>
- [Warp-up Report](/assets/Wrap-up_Report.pdf)
- 네이버 커넥트재단 부스트캠프 AI Tech 7기 NLP 과정으로 3주간 진행했습니다. (2024.11.11 ~ 2024.11.28)

## 프로젝트 수행 절차
- 아래의 일정표대로 프로젝트를 수행하였습니다.
![프로젝트 타임라인](/assets/project_time_line.png)

## 팀원과 역할
| 이름 | 역할 |
| --- | --- |
| 강전휘 <a href='https://github.com/kangjeonhwi'><img src='./assets/github.png' width=15 height=15 id='kjh'></img></a> | Generation 기반 CoT, Prompt Engineering |
| 박상준 <a href='https://github.com/bullmouse'><img src='./assets/github.png' width=15 height=15 id='psj'></img></a> | Model, KFold Cross Validation, Hyper-Parameter Tuning |
| 박준성 <a href='https://github.com/rasauq1122'><img src='./assets/github.png' width=15 height=15 id='pjs'></img></a>| OpenAI API를 사용한 Data Enhancing, CoT 실험 진행, 베이스라인 코드 모듈화, 앙상블 |
| 백승우 <a href='https://github.com/swbaek97'><img src='./assets/github.png' width=15 height=15 id='bsw'></img></a> | RAG (Sparse/Dense/Hybrid Retriever) |
| 서태영 <a href='https://github.com/sty0507'><img src='./assets/github.png' width=15 height=15 id='sty'></img></a> | EDA, Model(mistral, t3q), Hyper-Parameter Tuning |
| 이재백 <a href='https://github.com/Now100'><img src='./assets/github.png' width=15 height=15 id='ljb'></img></a> | EDA, Data Augmentation, Data Preprocessing(Label balancing), Zero-shot-CoT |

## 결과
- 최종 리더보드(private) 기준 Accuracy 점수 0.6460 달성
![리더보드 결과](/assets/leaderboard_score.png)

## 프로젝트 진행
- 프로젝트 진행하면서 적용한 내용들입니다.

|프로세스|설명|
| --- | --- |
| EDA | 데이터의 길이 분포 분석, 선다별 데이터 분포 분석|
| Augmentation | 외부 데이터 사용해서 데이터 증강 |
| Model | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, `T3Q-LLM/T3Q-LLM1-sft1.0-dpo1.0`, `upstage/SOLAR-10.7B-v1.0`, `yanolja/EEVE-Korean-10.8B-v1.0`, ... |
| RAG | Sparse Retriever(BM25), Dense Retriever(FAISS), Hybrid Retriever(Sparse + Dense) |
| Ensemble | Soft Voting, Hard Voting | 
| Etc | CoT, Zero-Shot-CoT |

## 개발 환경
| **Component** | **Specification** |
| --- | --- |
| **GPU** | NVIDIA Tesla V100 |
| **RAM** | 32 GB |
| **OS** | Linux |

## 프로젝트 구조
```bash
level2-nlp-generationfornlp-nlp-12-lv3
├── README.md
├── requirements.txt
├── assets
├── Experiments
├── Data
│   ├── EDA
│   └── Augmentation
├── RAG
│   └── wikipedia
├── config
└── src
    ├── main.py
    ├── manager
    ├── pipeline
    └── module
        └── ensemble.py

```
- `Experiments` : 프로젝트를 진행하면서 사용한 ipynb 파일 저장 디렉토리
- `Data` : 데이터 관련 작업(EDA, Augmentation)을 수행한 파일 저장 디렉토리
- `RAG` : RAG 관련 작업을 수행한 파일 저장 디렉토리
- `src` : 베이스 라인 코드 모듈화 파일 저장 디렉토리


## 코드 실행 방법
- **훈련 및 추론**
    ```bash
    # 경로 : ./
    python src/main.py --config {config_name} --train # 훈련
    python src/main.py --config {config_name} --inference # 추론
    python src/main.py --config {config_name} --do_both # 훈련 후 바로 추론
    ```
    
- **RAG**
    ```bash
    # 경로 : ./RAG/wikipedia/

    # 페이지 내용 모두 하나의 문서로 파싱
    python preprocess.py

    # 페이지 내용 문단별로 나눠서 각각 문서로 파싱, 같은 문서출신 문단은 id에 _{num}으로 구별
    python preprocess_paragraph.py

    # BM25S를 통해 Sparse Retriever Indexing
    python bm25.py

    # FAISS를 통해 Dense Retriever Indexing
    python dense_retriever.py
    ```