# Directory Architecture
```bash
├─main.py
├─trainer.py
├─inference.py
├─config
│  ├─prompts.py
│  ├─config.json
│  └─template.txt
├─utils
│  └─utils.py
└─README.md
```
- **`main.py`** : train 실행 파일
- **`trainer.py`** : train 코드 파일
- **`inference.py`** : inference 실행, 코드 파일
- **`config/config.json`** : learning_rate, epoch 등 train 파라미터 값 config 파일
- **`config/prompts.py`** : train에 사용되는 PROMPT 변수, TEMPLATE 변수 저장 파일
- **`utils/utils.py`** : json 파일, dataset 파일 불러오는 함수 저장 파일

# How to Use?
**학습 실행**
```bash
python main.py --config ./config/config
```
- **`--config`** : config 파일이 존재하는 폴더 경로 - .json을 뺀 파일 명
- eval accuracy 값이 출력됩니다.

**추론 실행**
```bash
python inference.py --config config/config --checkpoint ./output/checkpoint
```
- **`--config`** : config 파일이 존재하는 폴더 경로 - .json을 뺀 파일 명
- **`--checkpoint`** : checkpoint 폴더 경로 - train 실행시 output 파일 생성. output 파일 내에 있는 checkpoint 폴더

**config 파일**
- **`params`** : 학습에 사용되는 파라미터 값들
- **`settings/model_name`** : 학습에 사용할 model 이름
- **`settings/dataset`** : train, test 데이터 파일이 들어있는 폴더 경로