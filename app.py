import streamlit as st
from pdf_utils import extract_text_from_pdf, clean_text
from agent_core import run_pdf_agent

st.title("PDF Multi-Agent Analyzer")
st.write("PDF 파일을 업로드하면 여러 서브 에이전트가 협력하여 내용을 분석합니다.")

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file is not None:
    st.success(f"업로드 완료: {uploaded_file.name}")

    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)

    st.subheader("텍스트 정보")
    st.write(f"전체 글자 수: {len(cleaned_text)}")

    st.subheader("텍스트 미리보기")
    st.write(cleaned_text[:1500])

    if st.button("멀티 에이전트 분석 시작"):
        with st.spinner("서브 에이전트들이 PDF를 분석하는 중입니다..."):
            input_text = cleaned_text[:3000]
            final_result = run_pdf_agent(input_text)

        st.subheader("최종 분석 결과")
        st.write(final_result)
