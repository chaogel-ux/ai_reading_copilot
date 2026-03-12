import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_path = os.path.join(base_dir, "index", "review_index.faiss")
        meta_path = os.path.join(base_dir, "index", "review_texts.pkl")

        self.index = faiss.read_index(index_path)

        print("当前读取的 meta_path:", meta_path)
        print("当前文件大小:", os.path.getsize(meta_path))
        with open(meta_path, "rb") as f:
            self.records = pickle.load(f)

        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(ids[0]):
            record = self.records[idx]
            results.append({
                "score": float(scores[0][i]),
                "book_title": record.get("book_title"),
                "review_text": record.get("review_text")
            })

        return results


if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.search("人物刻画细腻，情感描写真实", top_k=3)

    for item in results:
        print(item)