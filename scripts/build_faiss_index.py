import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ========= 1. 修改保存路径为指定地址（核心变更）=========
# 源文件路径（保持不变，读取支持中文）
data_path = r"D:\workshixi\book_review_rag\data\raw\筛选high标签_书评.xlsx"
# 保存目录：指定的scripts路径（纯英文，避免FAISS兼容问题）
save_dir = r"D:\workshixi\ai_reading_copilot\index"
# 确保目录存在（添加递归创建，防止上级目录不存在）
os.makedirs(save_dir, mode=0o777, exist_ok=True)
print(f"✅ 保存目录：{save_dir}")

# 索引和元数据路径（拼接目标目录）
index_path = os.path.join(save_dir, "review_index.faiss")
meta_path = os.path.join(save_dir, "review_texts.pkl")

# ========= 2. 检查faiss版本（确保兼容）=========
print(f"✅ 当前faiss版本：{faiss.__version__}")
if not faiss.__version__.startswith("1.7"):
    print("⚠️ 建议安装faiss 1.7.4（最稳定版本）：pip install faiss-cpu==1.7.4")

# ========= 3. 检查源文件 =========
if not os.path.exists(data_path):
    print(f"❌ 源文件不存在：{data_path}")
    exit(1)

try:
    # ========= 4. 读取并清洗数据 =========
    df = pd.read_excel(data_path, engine="openpyxl")
    print(f"✅ 读取数据：{len(df)} 条")

    # 检查必要列
    required_cols = ["review_id", "book_title", "review_text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少列：{missing_cols}")
        exit(1)

    # 数据清洗
    df = df[required_cols].dropna()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]
    print(f"✅ 有效数据：{len(df)} 条")

    if len(df) == 0:
        print("❌ 无有效数据")
        exit(1)

    # 提前保存元数据（避免后续报错丢失）
    records = df.to_dict(orient="records")
    print("records数量：", len(records))

    with open(meta_path, "wb") as f:
        pickle.dump(records, f)

    print("pkl文件大小：", os.path.getsize(meta_path))
    print(f"✅ 元数据已提前保存：{meta_path}")
    # ========= 5. 生成Embedding =========
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = df["review_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"✅ Embedding形状：{embeddings.shape}")

    # ========= 6. 构建FAISS索引（浮点型，最基础兼容写法）=========
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 基础浮点索引，所有版本兼容
    index.add(embeddings)
    print(f"✅ 索引向量数：{index.ntotal}")

    # ========= 7. 保存索引（核心：路径字符串传参，适配所有版本）=========
    # 最基础的写法：传路径字符串，faiss官方推荐
    faiss.write_index(index, index_path)
    print(f"✅ FAISS索引保存成功：{index_path}")

    print("\n🎉 全部完成！")
    print(f"  - 索引文件：{index_path}")
    print(f"  - 元数据文件：{meta_path}")

# ========= 异常处理（终极兜底）=========
except PermissionError:
    print(f"❌ 权限不足！目标路径：{save_dir}")
    print("  解决方案：")
    print("  1. 以管理员身份运行Python脚本")
    print("  2. 检查目标路径是否被占用/只读")
    print("  3. 手动删除目标路径下的旧索引文件后重试")
except FileNotFoundError as e:
    print(f"❌ 文件不存在：{e}")
    print(f"  请检查：1. 源文件路径 {data_path} 2. 目标目录上级路径是否存在")
except Exception as e:
    print(f"❌ 运行失败：{str(e)}")
    # 终极备用方案：备份到桌面（100%有权限）
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop", "faiss_index_backup")
    os.makedirs(desktop_dir, exist_ok=True)
    # 保存元数据
    if 'records' in locals():
        with open(os.path.join(desktop_dir, "review_texts.pkl"), "wb") as f:
            pickle.dump(records, f)
    # 保存索引（如果已生成）
    if 'index' in locals():
        faiss.write_index(index, os.path.join(desktop_dir, "review_index.faiss"))
    print(f"✅ 已自动备份到桌面：{desktop_dir}")

exit(0)