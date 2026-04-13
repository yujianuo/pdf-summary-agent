import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.0-flash"


def generate_with_retry(prompt, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return f"요약 생성 실패: {str(e)}"


def summarize_chunk(chunk):
    prompt = f"""
다음은 PDF 문서의 일부 내용입니다.
핵심 내용을 간단하게 요약해 주세요.

내용:
{chunk}

출력 형식:
1. 요약
2. 핵심 포인트 3개
"""
    return generate_with_retry(prompt)


def final_summary(chunk_summaries):
    prompt = f"""
다음은 PDF 문서 각 부분의 요약입니다.
이를 바탕으로 전체 문서를 종합적으로 정리해 주세요.

내용:
{chunk_summaries}

출력 형식:
1. 전체 요약
2. 핵심 키워드 5개
3. 중요한 문장 3개
4. 주요 내용 정리
"""
    return generate_with_retry(prompt)
