import streamlit as st
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

st.set_page_config(page_title="AI Reading Copilot", page_icon="📚")

st.title("📚 AI Reading Copilot")
st.markdown("一个基于 **RAG + 大模型** 的智能书评生成助手")
st.markdown("---")

st.write("输入你的阅读感受，生成一段书评。")

book_title = st.text_input("请输入书名（可选）")

user_input = st.text_area("请输入你的阅读感受", height=150)

style = st.selectbox(
    "选择书评风格",
    ["理性分析", "情绪表达", "幽默吐槽"]
)

if st.button("生成书评"):
    if user_input.strip():
        with st.spinner("正在生成书评，请稍等..."):
            results = pipeline.retriever.search(user_input, top_k=3)
            result = pipeline.generate_review(user_input, style, book_title)

        st.subheader("检索到的参考书评")
        for i, item in enumerate(results, 1):
            st.write(f"{i}. 《{item['book_title']}》")
            st.write(item["review_text"])

        st.subheader("生成结果")
        st.write(result)
        st.code(result)
        score = pipeline.llm.generate(f"请从0-10分评价下面这段书评质量，并给出一句简短理由：\n\n{result}")

        st.subheader("AI评分")
        st.write(score)
    else:
        st.warning("请输入内容后再生成。")