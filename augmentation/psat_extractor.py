import fitz  # PyMuPDF 라이브러리
import re
import os
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.
    """
    try:
        # PDF 열기
        doc = fitz.open(pdf_path)
        extracted_text = []

        # 각 페이지의 텍스트 추출
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            extracted_text.append(text)

        return extracted_text
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def parse_questions_and_choices(text):
    """
    PDF에서 추출한 텍스트를 기반으로 지문, 질문, 보기, 선택지를 파싱합니다.
    """
    # 불필요한 문자열 제거
    text = re.sub(r"\d{4}년(?:도)?\s*(?:국가공무원\s*)?5급\s*(?:공채|공개경쟁채용|공채․외교관후보자\s*선발)(?:\s*등)?.*?언어논리영역(?:\s+[가-힣A-Z0-9])?\s*책형\s*\d+\s*쪽", "", text)

    questions = re.findall(r"\d+\.\s[^\]]+?(?=\?)", text)
    for i, question in enumerate(questions):
        questions[i] = re.sub(r"^\d+\.\s", "", question)
    text = re.sub(r"(\d+\.)\s[^\]]+?(?:\?)", r"\1", text)

    choices = re.findall(r"[①-⑤].+?(?=[①-⑤]|(?:\d+\.\s)|\[|\<실|\<보|\<사|※|$)", text)
    text = re.sub(r"[①-⑤].+?(?=[①-⑤]|(?:\d+\.\s)|\[|\<실|\<보|\<사|※|$)", "", text)

    tmp = []
    tmp_choices = []
    for i, choice in enumerate(choices):
        choice = re.sub(r"[①-⑤]", "", choice)
        tmp.append(choice)
        if i % 5 == 4:
            tmp_choices.append(tmp)
            tmp = []

    choices = tmp_choices

    text = re.sub(r"문(\d+)\.\s*～문(\d+)\.", r"\1～\2", text)

    paragraphs = re.findall(r"\d+\..+?(?=\d+\.\s|$)", text)
    text = re.sub(r"\d+\..+?(?=\d+\.\s|$)", "", text)
    
    for i, paragraph in enumerate(paragraphs):
        if re.findall(r"\[.*?\d+.*?[～~].*?\d+.*?\]", paragraph):
            parts = re.split(r"\[.*?\d+.*?[～~].*?\d+.*?\]", paragraph)
            paragraphs[i] = parts[0]
            paragraphs[i+1] = parts[1] + paragraphs[i+1]
            paragraphs[i+2] = parts[1] + paragraphs[i+2]
        
        paragraphs[i] = re.sub(r"\[.*?\d+.*?[～~].*?\d+.*?\]|\d+\.\s{3}", "", paragraph)
    
    return paragraphs, questions, choices

def extract_answer(answer_text):
    answer_list = [""]*40

    # 언어논리 부분의 정답만 추출
    answer = re.findall(r"언어논리[\s\S]*?(?=\d+년도)", answer_text)
    if not answer:
        raise ValueError("정답을 찾을 수 없습니다.")
    answer = re.sub(r"[가-힣]|㉮|㉯\s", "", answer[0])

    answers = answer.split("\n")
    answers = answers[:-1]
    
    answers = [ans for ans in answers if ans.isdigit()]

    for i in range(0, len(answers), 2):
        answer_list[int(answers[i]) - 1] = answers[i+1]
    
    return answer_list

if __name__ == "__main__":
    # PDF 파일 경로
    pdf_dir_path = "data/psat_problems"
    answer_dir_path = "data/psat_answers"
    
    file_list = sorted(os.listdir(pdf_dir_path))
    answer_file_list = sorted(os.listdir(answer_dir_path))
    
    data = []

    for i, (pdf_file, answer_file) in enumerate(zip(file_list, answer_file_list)):
        print(f"파일 {pdf_file}을 처리 중입니다.")
        # PDF에서 텍스트 추출
        pdf_path = os.path.join(pdf_dir_path, pdf_file)
        answer_path = os.path.join(answer_dir_path, answer_file)

        text_list = extract_text_from_pdf(pdf_path)
        answer_text_list = extract_text_from_pdf(answer_path)

        text = "".join(text_list).replace("\n", " ")
        answer_text = "".join(answer_text_list)

        # 텍스트를 파싱하여 지문, 질문, 보기, 선택지로 분리
        paragraphs, questions, choices = parse_questions_and_choices(text)

        # 정답을 추출
        answers = extract_answer(answer_text)
        
        for j, paragraph in enumerate(paragraphs):
            if len(paragraphs) != 40 or len(questions) != 40 or len(choices) != 40 or len(answers) != 40:
                print(len(paragraphs), len(questions), len(choices), len(answers))
                raise ValueError("지문, 질문, 보기, 정답의 개수가 일치하지 않습니다.")
            
            row = {"question_id": "psat" + str(i) + "_" + str(j + 1)}

            row["problems"] = f"{{'question': '{questions[j]}?', 'choices': {choices[j]}, 'answer': {answers[j]}}}"

            if "<보  기>" in paragraph or "<사  례>" in paragraph or "<보 기>" in paragraph:
                parts = re.split("<보  기>|<사  례>|<보 기>", paragraph)
                row["paragraph"] = parts[0]
                question_plus = "".join(parts[1:])
                if re.search(r"다음.+답하시오.", question_plus):
                    question_plus = re.sub(r"다음.+답하시오.+", "", question_plus)
                row["question_plus"] = question_plus
            else:
                row["paragraph"] = paragraph
                row["question_plus"] = ""

            if re.search(r"\.\s+문", row["paragraph"]):
                row["paragraph"] = re.sub(r"\.\s+문", "", row["paragraph"])

            if re.search(r"\d+\.\s+", row["paragraph"]):
                row["paragraph"] = re.sub(r"\d+\.\s+", "", row["paragraph"])

            if re.search(r"\.\s+문", row["problems"]):
                row["problems"] = re.sub(r"\.\s+문", "", row["problems"])
            
            if re.search(r"\s+문\s+", row["problems"]):
                row["problems"] = re.sub(r"\s+문\s+", "", row["problems"])
            
            if re.search(r"\.\s+문", row["question_plus"]):
                row["question_plus"] = re.sub(r"\.\s+문", "", row["question_plus"])
            
            ordered_row = {
                "question_id": row["question_id"],
                "paragraph": row["paragraph"],
                "problems": row["problems"],
                "question_plus": row["question_plus"]
            }
            
            data.append(ordered_row)

    df = pd.DataFrame(data)
    df.to_csv(f"data/psat_data.csv", index=False, encoding='utf-8-sig')
    print(f"파싱된 데이터가 psat_data.csv에 저장되었습니다.")
            