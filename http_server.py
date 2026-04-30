"""
Cato HTTP API Server (FastAPI)
POST /detect  - 检测单条文本
POST /batch   - 批量检测
GET  /health  - 健康检查
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import CatoModel, load_lexicon

model = CatoModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    lexicon = load_lexicon()
    if not model.load():
        raise RuntimeError("模型文件不存在，请先运行 main.py 训练模型")
    model.lexicon = lexicon
    print("模型加载完成")
    yield


app = FastAPI(title="Cato", version="0.1.0", lifespan=lifespan)


class DetectRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: list[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(req: DetectRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")
    return model.score_text(req.text)


@app.post("/batch")
async def batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is required")
    preds, probs = model.predict(req.texts)
    results = []
    for i, text in enumerate(req.texts):
        p = float(probs[i])
        results.append({
            "text": text[:200],
            "score": round(p, 4),
            "verdict": "违规" if preds[i] == 1 else "正常",
        })
    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8421)
