# AI Reading Copilot

一个基于 **RAG + 大语言模型** 的智能书评生成助手。

## 项目功能

- 输入阅读感受生成书评
- 基于 FAISS 向量检索相关书评
- 支持不同书评生成风格
- AI 自动评分生成书评质量
- Web 界面交互（Streamlit）

## 技术架构

RAG Pipeline；

用户输入  
↓  
Embedding 向量化  
↓  
FAISS 检索相关书评  
↓  
构建 Prompt  
↓  
LLM 生成书评  
↓  
AI 质量评分

## 技术栈

- Python
- FAISS
- Sentence Transformers
- GLM / 智谱 AI
- Streamlit

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```
启动应用：
```bash
streamlit run app.py
```
打开浏览器：
```bash
http://localhost:8501
```
项目结构：
```bash
ai_reading_copilot
│
├─ app.py
├─ rag_pipeline.py
│
├─ rag
│   ├─ retrieval.py
│   └─ llm.py
│
├─ index
│   ├─ review_index.faiss
│   └─ review_texts.pkl
│
├─ scripts
│   └─ build_faiss_index.py
│
└─ data
    └─ reviews.xlsx
```
