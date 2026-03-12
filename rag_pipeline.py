from rag.retrieval import Retriever
from rag.llm import LLM


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLM()

    def generate_review(self, user_input, style, book_title="", top_k=3):
        results = self.retriever.search(user_input, top_k=top_k)

        context = "\n".join(
            [f"{i + 1}.《{item['book_title']}》书评：{item['review_text']}" for i, item in enumerate(results)]
        )

        prompt = f"""
    请参考以下书评内容的表达方式、分析角度和语言风格，生成一段新的书评。

    书名：
    {book_title}
    
    用户输入：
    {user_input}
    
    生成风格：
    {style}
    
    参考书评：
    {context}

    要求：
    1. 语言自然，不要生硬拼接
    2. 看起来像真实用户写的书评
    3. 可以有情感表达和观点分析
    4. 不要直接照抄参考书评
    """

        return self.llm.generate(prompt)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    user_input = input("请输入你的阅读感受：")
    result = pipeline.generate_review(user_input)
    print(result)