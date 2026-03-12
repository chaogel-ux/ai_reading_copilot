from rag.retrieval import Retriever
from rag.llm import LLM


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLM()

    def generate_review(self, user_input, style, book_title="", length="中等", top_k=3):
        results = self.retriever.search(user_input, top_k=top_k)

        context = "\n".join(
            [f"{i + 1}.《{item['book_title']}》书评：{item['review_text']}" for i, item in enumerate(results)]
        )

        prompt = f"""
        你是一名擅长撰写书评的中文读者，请根据用户提供的书名、阅读感受，以及检索到的参考书评，生成一段新的书评。

        书名：
        {book_title}

        用户输入的阅读感受：
        {user_input}

        生成风格：
        {style}

        输出长度：
        {length}

        参考书评：
        {context}

        写作要求：
        1. 优先结合用户输入的阅读感受来组织内容。
        2. 可以参考“参考书评”中的分析角度、表达方式或评价思路，但不要直接复制原句。
        3. 如果用户输入较短，可以基于书名和参考书评适度补充合理的观点。
        4. 语言要自然，像真实读者写的书评，不要像机器说明文。
        5. 根据所选风格调整表达方式。
        6. 根据所选长度控制篇幅。
        7. 输出直接给出书评正文，不要加“以下是书评”之类的说明。
        """

        return self.llm.generate(prompt)


if __name__ == "__main__":
    pipeline = RAGPipeline()
    user_input = input("请输入你的阅读感受：")
    result = pipeline.generate_review(user_input)
    print(result)