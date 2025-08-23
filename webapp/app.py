from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from services.video_service import VideoService
from services.llm_service import LLMService
from config import settings
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

app = FastAPI(title="Video LLM Reasoning System")

# Setup static files and templates
os.makedirs("webapp/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")

# Initialize services
video_service = VideoService()
llm_service = LLMService()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/api/process_video")
async def process_video(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")

        file_ext = file.filename.split(".")[-1]
        video_bytes = await file.read()

        video_tree = video_service.process_video_file(video_bytes, file_ext)
        representative_frames = video_service.get_representative_frames(video_tree)

        return {
            "message": "Video processed successfully",
            "num_frames": len(video_tree.frames),
            "num_nodes": len(video_tree.nodes),
            "representative_frames": [
                {
                    "timestamp": frame.timestamp,
                    "node_id": next(
                        n.id for n in video_tree.nodes.values()
                        if frame.index in n.frame_indices
                    )
                }
                for frame in representative_frames
            ]
        }
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_video(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")

        file_ext = file.filename.split(".")[-1]
        video_bytes = await file.read()

        video_tree = video_service.process_video_file(video_bytes, file_ext)
        summary = llm_service.generate_summary(video_tree)

        return {
            "summary": summary,
            "num_frames": len(video_tree.frames)
        }
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/answer")
async def answer_question(file: UploadFile = File(...), question: str = ""):
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        file_ext = file.filename.split(".")[-1]
        video_bytes = await file.read()

        video_tree = video_service.process_video_file(video_bytes, file_ext)
        answer = llm_service.answer_question(video_tree, question)

        return {
            "answer": answer,
            "question": question
        }
    except Exception as e:
        logger.error(f"Question answering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)