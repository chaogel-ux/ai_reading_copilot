from zhipuai import ZhipuAI


class LLM:

    def __init__(self):
        self.client = ZhipuAI(
            api_key="fba94c0d370d4a4d9798b705c1783a86.YQ92foPNNRyQcELP"
        )

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是一名专业的文学书评作者"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content