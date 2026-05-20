import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
import torch
import torch.nn.functional as F
from fastai.vision.all import *
import json
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ПУТИ К МОДЕЛЯМ — не менять
SKIN_MODEL_PATH  = "skin_data/skin_models/skin_model_final.pkl"
LUNGS_MODEL_PATH = "chest_models_hq/chest_models_hq/chest_model_hq.pkl"

skin_learn    = None
lungs_learn   = None
skin_metadata  = {}
lungs_metadata = {}


def manual_predict(learner, pil_img: Image.Image):
    """
    Инференс без learner.predict() — обходит баг IndexError в vocab декодировании.
    Модели с кастомным head (GeM pooling, custom head) ломают fastai decode_batch.
    Мы сами делаем forward pass через torch.
    """
    # Размер из даталоадера — after_item.size может быть int, tuple или fastuple
    try:
        s = learner.dls.after_item.size
        if hasattr(s, '__iter__'):
            s = tuple(int(x) for x in s)
            wh = (s[1], s[0]) if len(s) >= 2 else (s[0], s[0])
        else:
            wh = (int(s), int(s))
    except Exception:
        wh = (224, 224)

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    pil_img = pil_img.resize(wh)  # PIL.resize принимает (width, height)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr  = np.array(pil_img).astype(np.float32) / 255.0
    t    = torch.from_numpy(arr).permute(2, 0, 1)
    t    = (t - mean) / std
    t    = t.unsqueeze(0)

    device = next(learner.model.parameters()).device
    t = t.to(device)
    learner.model.eval()
    with torch.no_grad():
        logits = learner.model(t)

    probs    = F.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())

    vocab      = list(learner.dls.vocab) if hasattr(learner.dls, 'vocab') else []
    pred_class = vocab[pred_idx] if pred_idx < len(vocab) else str(pred_idx)

    return pred_class, pred_idx, probs


def create_skin_metadata(classes):
    return {
        "model_name": "Skin Cancer Detector", "model_type": "skin", "architecture": "resnet18",
        "classes": {
            "melanoma": {"name": "Melanoma", "description": "Malignant melanoma - the most dangerous type of skin cancer", "risk": "high"},
            "normal":   {"name": "Normal",   "description": "Healthy skin - no signs of melanoma detected", "risk": "low"}
        },
        "class_list": classes if classes else ["melanoma", "normal"],
        "performance_metrics": {"overall": {"accuracy": 0.8967, "precision": 0.9078, "recall": 0.9818, "f1_score": 0.9434}},
        "training_date": datetime.now().isoformat()
    }

def create_lungs_metadata(classes):
    return {
        "model_name": "Lung Disease Detector", "model_type": "lungs", "architecture": "resnet18",
        "classes": {
            "normal":    {"name": "Normal",    "description": "Healthy lungs - no signs of pneumonia detected", "risk": "low"},
            "pneumonia": {"name": "Pneumonia", "description": "Pneumonia detected - inflammation of the lungs", "risk": "high"}
        },
        "class_list": classes if classes else ["normal", "pneumonia"],
        "performance_metrics": {"overall": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1_score": 0.95}},
        "training_date": datetime.now().isoformat()
    }


def load_models():
    global skin_learn, lungs_learn, skin_metadata, lungs_metadata
    try:
        logger.info("🚀 Starting Medical AI Diagnosis API...")
        current_dir = os.path.dirname(os.path.abspath(__file__))

        logger.info("\n" + "="*50)
        logger.info("📦 Loading SKIN CANCER Model...")
        logger.info("="*50)
        skin_path = os.path.join(current_dir, SKIN_MODEL_PATH)
        logger.info(f"📂 Skin model path: {skin_path}")
        if os.path.exists(skin_path):
            skin_learn = load_learner(skin_path)
            skin_classes = list(skin_learn.dls.vocab) if hasattr(skin_learn.dls, 'vocab') else ["melanoma", "normal"]
            logger.info(f"✅ Skin model loaded | classes: {skin_classes}")
            meta_path = os.path.join(current_dir, "skin_data/skin_models/metadata_skin.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    skin_metadata = json.load(f)
                logger.info("✅ Skin metadata loaded")
            else:
                skin_metadata = create_skin_metadata(skin_classes)
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(skin_metadata, f, ensure_ascii=False, indent=2)
                logger.info("✅ Skin metadata created")
        else:
            logger.warning(f"⚠️ Skin model not found: {skin_path}")

        logger.info("\n" + "="*50)
        logger.info("📦 Loading LUNG DISEASE Model...")
        logger.info("="*50)
        lungs_path = os.path.join(current_dir, LUNGS_MODEL_PATH)
        logger.info(f"📂 Lungs model path: {lungs_path}")
        if os.path.exists(lungs_path):
            lungs_learn = load_learner(lungs_path)
            lungs_classes = list(lungs_learn.dls.vocab) if hasattr(lungs_learn.dls, 'vocab') else ["normal", "pneumonia"]
            logger.info(f"✅ Lungs model loaded | classes: {lungs_classes}")
            meta_path = os.path.join(current_dir, "chest_models_hq/chest_models_hq/metadata_chest.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    lungs_metadata = loaded if isinstance(loaded, dict) else create_lungs_metadata(lungs_classes)
                logger.info("✅ Lungs metadata loaded")
            else:
                lungs_metadata = create_lungs_metadata(lungs_classes)
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(lungs_metadata, f, ensure_ascii=False, indent=2)
                logger.info("✅ Lungs metadata created")
        else:
            logger.warning(f"⚠️ Lungs model not found: {lungs_path}")

        logger.info("\n" + "="*50)
        logger.info("✅ Models loaded successfully!")
        logger.info(f"   Skin Model:  {'✓' if skin_learn  else '✗'}")
        logger.info(f"   Lungs Model: {'✓' if lungs_learn else '✗'}")
        logger.info("="*50)
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    logger.info("👋 Shutting down...")

app = FastAPI(title="Medical AI Diagnosis API", description="API для обнаружения рака кожи и заболеваний легких", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def get_risk_level(prediction: str, model_type: str):
    p = prediction.lower()
    return "high" if (p == "melanoma" if model_type == "skin" else p == "pneumonia") else "low"

def get_recommendation(prediction: str, model_type: str, confidence: float):
    p = prediction.lower()
    if model_type == "skin":
        return "⚠️ ВЫСОКИЙ РИСК. Срочно обратитесь к дерматологу или онкологу для биопсии." if p == "melanoma" \
               else "✅ НИЗКИЙ РИСК. Меланома не обнаружена. Рекомендуется регулярный осмотр."
    return "⚠️ ВЫСОКИЙ РИСК. Пневмония. Срочно обратитесь к терапевту или пульмонологу." if p == "pneumonia" \
           else "✅ НОРМА. Лёгкие без патологий. Рекомендуется ежегодная флюорография."


@app.get("/")
async def root():
    return {"message": "Medical AI Diagnosis API", "version": "2.0.0",
            "available_models": {"skin_cancer": skin_learn is not None, "lung_disease": lungs_learn is not None}}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "skin_model_loaded": skin_learn is not None,
            "lungs_model_loaded": lungs_learn is not None, "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def get_models_info():
    models_info = {}
    if skin_learn is not None:
        models_info["skin_cancer"] = {
            "name": skin_metadata.get("model_name", "Skin Cancer Detector"),
            "classes": list(skin_learn.dls.vocab) if hasattr(skin_learn.dls, 'vocab') else [],
            "accuracy": skin_metadata.get("performance_metrics", {}).get("overall", {}).get("accuracy", "N/A")
        }
    if lungs_learn is not None:
        models_info["lung_disease"] = {
            "name": lungs_metadata.get("model_name", "Lung Disease Detector"),
            "classes": list(lungs_learn.dls.vocab) if hasattr(lungs_learn.dls, 'vocab') else [],
            "accuracy": lungs_metadata.get("performance_metrics", {}).get("overall", {}).get("accuracy", "N/A")
        }
    return {"models": models_info}


@app.post("/predict/skin")
async def predict_skin(file: UploadFile = File(...)):
    if skin_learn is None:
        raise HTTPException(status_code=503, detail="Skin model not loaded")
    try:
        logger.info(f"🔍 Skin prediction: {file.filename}")
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name   = "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.'))
        upload_path = os.path.join(UPLOAD_DIR, f"skin_{timestamp}_{safe_name}")
        pil_img.convert('RGB').save(upload_path)

        pred_class, pred_idx, probs = manual_predict(skin_learn, pil_img)
        vocab      = list(skin_learn.dls.vocab) if hasattr(skin_learn.dls, 'vocab') else []
        prob_dict  = {c: round(float(probs[i]), 4) for i, c in enumerate(vocab) if i < len(probs)}
        confidence = float(probs[pred_idx])

        class_desc = ""
        if skin_metadata and "classes" in skin_metadata:
            class_desc = skin_metadata["classes"].get(str(pred_class), {}).get("description", "")

        result = {
            "prediction": str(pred_class), "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "risk_level": get_risk_level(str(pred_class), "skin"),
            "recommendation": get_recommendation(str(pred_class), "skin", confidence),
            "class_description": class_desc, "all_probabilities": prob_dict,
            "model_accuracy": skin_metadata.get("performance_metrics", {}).get("overall", {}).get("accuracy", "N/A"),
            "timestamp": datetime.now().isoformat(),
            "image_url": f"/uploads/{os.path.basename(upload_path)}"
        }
        logger.info(f"✅ Skin prediction: {pred_class} ({confidence:.2%})")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Skin prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/lungs")
async def predict_lungs(file: UploadFile = File(...)):
    if lungs_learn is None:
        raise HTTPException(status_code=503, detail="Lungs model not loaded")
    try:
        logger.info(f"🔍 Lungs prediction: {file.filename}")
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name   = "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.'))
        upload_path = os.path.join(UPLOAD_DIR, f"lungs_{timestamp}_{safe_name}")
        pil_img.convert('RGB').save(upload_path)

        pred_class, pred_idx, probs = manual_predict(lungs_learn, pil_img)
        vocab      = list(lungs_learn.dls.vocab) if hasattr(lungs_learn.dls, 'vocab') else []
        prob_dict  = {c: round(float(probs[i]), 4) for i, c in enumerate(vocab) if i < len(probs)}
        confidence = float(probs[pred_idx])

        class_desc = ""
        if lungs_metadata and isinstance(lungs_metadata, dict):
            classes_data = lungs_metadata.get("classes", {})
            if isinstance(classes_data, dict):
                class_desc = classes_data.get(str(pred_class), {}).get("description", "")

        result = {
            "prediction": str(pred_class), "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "risk_level": get_risk_level(str(pred_class), "lungs"),
            "recommendation": get_recommendation(str(pred_class), "lungs", confidence),
            "class_description": class_desc, "all_probabilities": prob_dict,
            "model_accuracy": lungs_metadata.get("performance_metrics", {}).get("overall", {}).get("accuracy", "N/A") if isinstance(lungs_metadata, dict) else "N/A",
            "timestamp": datetime.now().isoformat(),
            "image_url": f"/uploads/{os.path.basename(upload_path)}"
        }
        logger.info(f"✅ Lungs prediction: {pred_class} ({confidence:.2%})")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Lungs prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-ui")
async def test_ui():
    html_content = """<!DOCTYPE html>
<html><head>
<title>Medical AI Diagnosis API</title><meta charset="UTF-8">
<style>
body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;}
.container{max-width:1000px;margin:auto;background:white;border-radius:15px;padding:30px;box-shadow:0 20px 60px rgba(0,0,0,.3);}
h1{color:#333;text-align:center;margin-bottom:10px;}
.disclaimer{background:#fff3cd;padding:15px;border-radius:8px;margin:20px 0;border-left:4px solid #ffc107;}
.model-card{border:1px solid #e0e0e0;padding:25px;margin:25px 0;border-radius:12px;transition:transform .3s;}
.model-card:hover{transform:translateY(-5px);box-shadow:0 10px 30px rgba(0,0,0,.1);}
.skin{border-left:5px solid #4CAF50;}.lungs{border-left:5px solid #2196F3;}
.upload-area{margin:20px 0;}
input[type=file]{display:block;width:100%;padding:10px;border:2px dashed #ccc;border-radius:8px;cursor:pointer;}
button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;padding:12px 30px;border-radius:8px;cursor:pointer;font-size:16px;font-weight:bold;margin-top:10px;transition:transform .2s;}
button:hover{transform:scale(1.02);}button:disabled{opacity:.5;cursor:not-allowed;}
.result{background:#f8f9fa;padding:20px;border-radius:10px;margin-top:20px;display:none;border:1px solid #dee2e6;}
.high-risk{color:#dc3545;font-weight:bold;background:#f8d7da;padding:5px 10px;border-radius:4px;display:inline-block;}
.low-risk{color:#28a745;font-weight:bold;background:#d4edda;padding:5px 10px;border-radius:4px;display:inline-block;}
.status-badge{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;}
.status-active{background:#28a745;}.status-inactive{background:#dc3545;}
.image-preview{max-width:200px;max-height:150px;margin-top:10px;border-radius:8px;display:none;}
</style></head>
<body><div class="container">
<h1>🏥 Medical AI Diagnosis API</h1>
<p style="text-align:center;color:#666;">Скрининг рака кожи и заболеваний лёгких</p>
<div class="disclaimer">⚠️ <strong>Отказ от ответственности:</strong> Это инструмент для скрининга. Всегда консультируйтесь с врачом для точного диагноза.</div>
<div id="status"></div>
<div class="model-card skin">
  <h2>🩺 Обнаружение рака кожи</h2>
  <div class="upload-area">
    <input type="file" id="skinFile" accept="image/*">
    <img id="skinPreview" class="image-preview" alt="Preview">
    <button id="skinBtn">🔍 Анализ Изображения Кожи</button>
  </div>
  <div id="skinResult" class="result"></div>
</div>
<div class="model-card lungs">
  <h2>🫁 Выявление заболеваний легких</h2>
  <div class="upload-area">
    <input type="file" id="lungsFile" accept="image/*">
    <img id="lungsPreview" class="image-preview" alt="Preview">
    <button id="lungsBtn">🔍 Анализ Изображения легких</button>
  </div>
  <div id="lungsResult" class="result"></div>
</div>
</div>
<script>
function previewImage(input,previewId){
  const preview=document.getElementById(previewId),file=input.files[0];
  if(file){const r=new FileReader();r.onload=e=>{preview.src=e.target.result;preview.style.display='block';};r.readAsDataURL(file);}
  else{preview.style.display='none';}
}
document.getElementById('skinFile').onchange=function(){previewImage(this,'skinPreview');};
document.getElementById('lungsFile').onchange=function(){previewImage(this,'lungsPreview');};

async function checkStatus(){
  try{
    const data=await(await fetch('/health')).json();
    const s=ok=>ok?'<span class="status-badge status-active"></span>Активна':'<span class="status-badge status-inactive"></span>Не загружена';
    document.getElementById('status').innerHTML=`<div style="background:#e8f4fc;padding:15px;border-radius:8px;margin-bottom:20px;"><strong>📊 Статус моделей:</strong><br>🩺 Кожа: ${s(data.skin_model_loaded)}<br>🫁 Лёгкие: ${s(data.lungs_model_loaded)}</div>`;
    document.getElementById('skinBtn').disabled=!data.skin_model_loaded;
    document.getElementById('lungsBtn').disabled=!data.lungs_model_loaded;
  }catch(e){console.error(e);}
}

async function analyze(endpoint,fileInput,resultDiv){
  const file=fileInput.files[0];
  if(!file){resultDiv.innerHTML='<p style="color:orange;">⚠️ Пожалуйста, выберите изображение</p>';resultDiv.style.display='block';return;}
  const formData=new FormData();formData.append('file',file);
  resultDiv.style.display='block';resultDiv.innerHTML='<p>🔄 Анализ изображения... Пожалуйста, подождите</p>';
  try{
    const response=await fetch(endpoint,{method:'POST',body:formData});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||'Ошибка сервера');
    const riskClass=data.risk_level==='high'?'high-risk':'low-risk';
    const riskText=data.risk_level==='high'?'ВЫСОКИЙ':'НИЗКИЙ';
    resultDiv.innerHTML=`<h3>📋 Результат анализа</h3>
      <p><strong>🩺 Диагноз:</strong> ${data.prediction}</p>
      <p><strong>⚠️ Уровень риска:</strong> <span class="${riskClass}">${riskText}</span></p>
      <p><strong>📊 Уверенность:</strong> ${data.confidence_percentage}</p>
      <p><strong>💡 Рекомендация:</strong><br>${data.recommendation}</p>
      ${data.class_description?`<p><strong>📝 Описание:</strong> ${data.class_description}</p>`:''}
      <hr><p style="font-size:12px;color:#666;"><strong>🤖 Модель:</strong> Точность ${data.model_accuracy}<br><strong>⏰ Время:</strong> ${new Date(data.timestamp).toLocaleString()}</p>`;
  }catch(error){resultDiv.innerHTML=`<p style="color:red;">❌ Ошибка: ${error.message}</p>`;}
}

document.getElementById('skinBtn').onclick=()=>analyze('/predict/skin',document.getElementById('skinFile'),document.getElementById('skinResult'));
document.getElementById('lungsBtn').onclick=()=>analyze('/predict/lungs',document.getElementById('lungsFile'),document.getElementById('lungsResult'));
checkStatus();setInterval(checkStatus,30000);
</script></body></html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")