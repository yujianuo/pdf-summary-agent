from dotenv import load_dotenv
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent
import re

load_dotenv()

MODEL_NAME = "google_genai:gemini-2.0-flash"


# -----------------------------
# Sub-agents
# -----------------------------
reader_graph = create_agent(
    model=MODEL_NAME,
    tools=[],
    system_prompt="""
You are Reader Agent.

The extracted PDF text is already provided by the supervisor.
Do not ask the user for the PDF again.

Your job:
1. Read the provided PDF text.
2. Identify the document topic.
3. Organize the content into main sections.
4. Briefly explain each section.

Return:
- Document topic
- Main sections
- Short explanation of each section
"""
)

reader_subagent = CompiledSubAgent(
    name="reader-agent",
    description="Reads extracted PDF text and identifies the topic and structure.",
    runnable=reader_graph,
)

summary_graph = create_agent(
    model=MODEL_NAME,
    tools=[],
    system_prompt="""
You are Summary Agent.

The extracted PDF text is already provided by the supervisor.
Do not ask the user for the PDF again.

Your job:
1. Write a short summary of the document.
2. Write a more detailed summary in bullet points.

Return:
- Short summary
- Detailed summary
"""
)

summary_subagent = CompiledSubAgent(
    name="summary-agent",
    description="Summarizes the PDF into short and detailed summaries.",
    runnable=summary_graph,
)

insight_graph = create_agent(
    model=MODEL_NAME,
    tools=[],
    system_prompt="""
You are Insight Agent.

The extracted PDF text is already provided by the supervisor.
Do not ask the user for the PDF again.

Your job:
1. Extract 5 keywords.
2. Extract 3 important sentences.
3. Identify major takeaways.

Return:
- Keywords
- Important sentences
- Key takeaways
"""
)

insight_subagent = CompiledSubAgent(
    name="insight-agent",
    description="Extracts keywords, important sentences, and key takeaways from the PDF.",
    runnable=insight_graph,
)


# -----------------------------
# Supervisor Agent
# -----------------------------
pdf_team_agent = create_deep_agent(
    model=MODEL_NAME,
    system_prompt="""
You are the supervisor of a PDF analysis agent team.

Important rules:
- The user has already provided the extracted PDF text.
- Never ask the user to provide the PDF text again.
- You must use the given text as the source material.
- You must delegate work to the subagents.

Required workflow:
1. Use write_todos to create a short plan.
2. Send the provided PDF text to reader-agent for structure analysis.
3. Send the provided PDF text to summary-agent for summarization.
4. Send the provided PDF text to insight-agent for keywords, important sentences, and takeaways.
5. Merge the results into one final answer.

Final answer must be in Korean and follow exactly this format:
1. 전체 요약
2. 핵심 키워드
3. 중요한 문장
4. 주요 내용 정리
""",
    subagents=[reader_subagent, summary_subagent, insight_subagent],
    name="pdf-supervisor-agent",
)


# -----------------------------
# Local fallback functions
# -----------------------------
def split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?。！？])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def extract_keywords_simple(text: str, top_n=5):
    words = re.findall(r"[A-Za-z가-힣]{2,}", text.lower())
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "have", "will",
        "are", "was", "were", "into", "using", "used", "also", "than", "then",
        "page", "python", "docker", "agent", "summary", "text", "pdf", "내용",
        "주차", "오늘", "이번", "설명", "이해", "수업", "학습", "있다", "한다"
    }

    freq = {}
    for w in words:
        if w not in stopwords and len(w) >= 2:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def local_fallback_analysis(text: str) -> str:
    sentences = split_sentences(text)

    short_summary = ""
    if sentences:
        short_summary = " ".join(sentences[:3])

    keywords = extract_keywords_simple(text, top_n=5)

    important_sentences = sentences[:3] if len(sentences) >= 3 else sentences

    takeaways = []
    if "docker" in text.lower():
        takeaways.append("Docker 관련 개념과 실행 환경 구성이 중요한 학습 포인트로 보입니다.")
    if "python" in text.lower():
        takeaways.append("Python 라이브러리 및 패키지 사용에 대한 이해가 핵심 내용으로 보입니다.")
    if "api" in text.lower():
        takeaways.append("API 활용 및 설계 원리에 대한 내용이 포함되어 있습니다.")
    if not takeaways:
        takeaways = [
            "문서의 주요 개념을 중심으로 핵심 내용을 정리할 필요가 있습니다.",
            "중요 개념과 예시를 함께 보면 문서 이해에 도움이 됩니다.",
            "전체 흐름을 파악한 뒤 세부 항목을 정리하는 방식이 효과적입니다."
        ]

    detailed_points = []
    for s in sentences[:5]:
        detailed_points.append(f"- {s}")

    result = f"""
1. 전체 요약
{short_summary if short_summary else "문서의 앞부분을 바탕으로 핵심 내용을 요약했습니다."}

2. 핵심 키워드
{", ".join(keywords) if keywords else "키워드 추출 실패"}

3. 중요한 문장
{chr(10).join([f"- {s}" for s in important_sentences]) if important_sentences else "- 중요한 문장을 찾지 못했습니다."}

4. 주요 내용 정리
{chr(10).join(detailed_points) if detailed_points else "- 주요 내용을 정리하지 못했습니다."}

추가 분석 포인트:
{chr(10).join([f"- {t}" for t in takeaways])}
"""
    return result.strip()


# -----------------------------
# Main run function
# -----------------------------
def run_pdf_agent(extracted_text: str) -> str:
    prompt = f"""
The extracted PDF text is already provided below.
You must analyze this text directly.
Do not ask for the PDF text again.

<PDF_TEXT>
{extracted_text}
</PDF_TEXT>

Please follow the required workflow:
- delegate to reader-agent
- delegate to summary-agent
- delegate to insight-agent
- then produce the final answer in Korean

Required final format:
1. 전체 요약
2. 핵심 키워드
3. 중요한 문장
4. 주요 내용 정리
"""

    try:
        result = pdf_team_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", "")
            if isinstance(content, str) and content.strip():
                return content
            return str(content)

        return str(result)

    except Exception as e:
        error_text = str(e)

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "quota" in error_text.lower():
            fallback_result = local_fallback_analysis(extracted_text)
            return (
                "Gemini API quota exceeded, so local fallback analysis was used.\n\n"
                + fallback_result
            )

        return f"분석 중 오류가 발생했습니다:\n{error_text}"
