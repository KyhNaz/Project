from PIL import Image
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile,Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import  HTMLResponse

from transformers import CLIPProcessor, CLIPModel

from typing import List

from sqlalchemy.orm import Session

from database import Base, SessionLocal,engine

from models import categoreyes

from io import BytesIO
import io
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/img", StaticFiles(directory="static/img"), name="img")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        # 마지막에 무조건 닫음
        db.close()

Base.metadata.create_all(bind=engine)

labels = ["Human", "Animal", "Food", "Document", "Landscape"]

upload_dirs = ['human', 'animal', 'food', 'document', 'landscape']
for upload_dir in upload_dirs:
    Path(f'./uploads/{upload_dir}').mkdir(parents=True, exist_ok=True)

@app.post("/predict")
async def predict(files: List[UploadFile],db: Session = Depends(get_db)):
   
    results = []
    for file in files:
        try:
            # 이미지 업로드 및 처리
            image = Image.open(BytesIO(await file.read()))
            image_in_bytes = io.BytesIO()
            image.save(image_in_bytes, format='PNG')
            
            # CLIP 모델 입력 생성
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
            
            # 모델 실행
            outputs = model(**inputs)
            
            # 결과 처리
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            result = {"file_name": file.filename, "probs": probs.tolist()}
        
            # 가장 확률이 높은 카테고리를 선택
            categories = ["Human","Animal","Food","Document","Landscape"]
            category = categories[probs.argmax().item()]

            new_image = categoreyes(data=image_in_bytes.getvalue(), category=category, filename=file.filename)
            db.add(new_image)
            db.commit()
            
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
 
    
    return {"results":results}

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi import File, UploadFile

@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    
    return {"message": "Files uploaded successfully."}
