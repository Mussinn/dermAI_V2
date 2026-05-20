"""
chest_pipeline.py — Скачивание данных + Обучение классификатора лёгких
═══════════════════════════════════════════════════════════════════════
Источники (без авторизации, публичные):
  1. Montgomery County CXR (NIH/NLM) — нормальные лёгкие (138 снимков)
  2. Shenzhen Hospital CXR (NIH/NLM) — нормальные + туберкулёз → берём нормальные
  3. Indiana University CXR (OpenI/NLM) — смешанный набор
  4. VinBigData open sample — пневмония
  5. NIH ChestX-ray14 mini-set — пневмония + normal (публичный)

Структура:
    chest_data/
        pneumonia/    ← рентгены с пневмонией
        normal/       ← здоровые рентгены
        .done_pneumonia
        .done_normal
    chest_models_hq/
        chest_model_hq.pkl
        metadata_chest.json

Запуск:
    python chest_pipeline.py                   # скачать + обучить
    python chest_pipeline.py --download-only
    python chest_pipeline.py --train-only
    python chest_pipeline.py --force-download
    python chest_pipeline.py --predict img.jpg
    python chest_pipeline.py --max 800
"""

import os, sys, json, argparse, warnings, random, time, zipfile, tarfile, csv
import urllib.request, urllib.parse, urllib.error
import shutil
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────
CFG = {
    "DATA_DIR":          "chest_data",
    "MODELS_DIR":        "chest_models_hq",
    "EXPORT_NAME":       "chest_model_hq.pkl",
    "META_NAME":         "metadata_chest.json",
    "CACHE_DIR":         ".chest_cache",

    "CLASSES":           ["normal", "pneumonia"],
    "POSITIVE_CLASS":    "pneumonia",

    "MAX_PER_CLASS":     1000,
    "DELAY":             0.1,

    # Обучение — X-ray специфика
    "IMG_SIZE":          224,
    "BATCH_SIZE":        8,
    "EPOCHS_HEAD":       3,
    "EPOCHS_UNFREEZE":   5,
    "EPOCHS_FULL":       10,
    "VALID_PCT":         0.15,
    "SEED":              42,
    "BASE_LR":           5e-4,
    "LABEL_SMOOTHING":   0.03,   # меньше для X-ray (более чёткие границы)
    "DROPOUT":           0.40,
    "GRAD_CLIP":         1.0,
    "EARLY_STOP_PAT":    6,
    "TTA_N":             4,
    "FLIP_VERT":         False,  # ВАЖНО: рентгены не переворачивать!
    "MAX_ROTATE":        8.0,
    "MAX_WARP":          0.03,
}

FLAG_PNEUMONIA = os.path.join(CFG["DATA_DIR"], ".done_pneumonia")
FLAG_NORMAL    = os.path.join(CFG["DATA_DIR"], ".done_normal")


# ─────────────────────────────────────────────
#  УТИЛИТЫ
# ─────────────────────────────────────────────
def banner(text, char="═"):
    w = 62
    print(f"\n{char*w}\n  {text}\n{char*w}")

def log(msg, tag="INFO"):
    icons = {"INFO":"ℹ️ ","OK":"✅","WARN":"⚠️ ","ERR":"❌","DL":"⬇️ ","TRAIN":"🧠"}
    print(f"  {icons.get(tag,'  ')} {msg}", flush=True)

def ensure_dirs():
    for cls in CFG["CLASSES"]:
        os.makedirs(os.path.join(CFG["DATA_DIR"], cls), exist_ok=True)
    os.makedirs(CFG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
    log("Directory structure created", "OK")

def count_images(directory):
    if not os.path.exists(directory): return 0
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
    return sum(1 for f in Path(directory).iterdir() if f.suffix.lower() in exts)

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def gpu_info():
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}", "OK")
    else:
        log("GPU not available — CPU mode", "WARN")

def _fetch_url(url, dest, desc="", timeout=60, retries=3):
    """Скачивает файл с повторными попытками."""
    dest = Path(dest)
    if dest.exists(): return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dest) + ".tmp"
    for attempt in range(1, retries+1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"MedAI/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length",0))
                done  = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1024*256)
                        if not chunk: break
                        f.write(chunk); done+=len(chunk)
                        if total>0:
                            pct = done/total*100
                            print(f"\r    {desc} {pct:.0f}% ({done/1e6:.1f}/{total/1e6:.1f} MB)",
                                  end="", flush=True)
            print()
            os.rename(tmp, dest); return True
        except Exception as e:
            log(f"Attempt {attempt}: {e}", "WARN")
            if os.path.exists(tmp): os.remove(tmp)
            if attempt < retries: time.sleep(2**attempt)
    return False


# ─────────────────────────────────────────────
#  ИСТОЧНИК 1: MONTGOMERY COUNTY (нормальные лёгкие)
# ─────────────────────────────────────────────
MONTGOMERY_BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/"
    "Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/"
)
# 138 снимков: MCUCXR_0001_0.png ... MCUCXR_0138_0.png
# _0 = нормальные, _1 = туберкулёз (берём только _0)
MONTGOMERY_NORMAL = [f"MCUCXR_{str(i).zfill(4)}_0.png" for i in range(1, 81)]
# Первые 80 нормальных снимков достаточно


def download_montgomery_normal(dest_dir, max_count):
    """Скачивает нормальные рентгены из Montgomery County CXR dataset."""
    log("Source 1: Montgomery County CXR (normal lungs)", "DL")
    ok = 0
    for fname in MONTGOMERY_NORMAL:
        if ok >= max_count: break
        dest = Path(dest_dir) / fname
        url  = MONTGOMERY_BASE + fname
        if dest.exists(): ok+=1; continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"MedAI/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            with open(dest, "wb") as f: f.write(data)
            ok+=1; time.sleep(CFG["DELAY"])
        except: pass
        if ok % 20 == 0 and ok > 0:
            log(f"  Montgomery normal: {ok} downloaded", "INFO")
    log(f"Montgomery: {ok} normal images", "OK")
    return ok


# ─────────────────────────────────────────────
#  ИСТОЧНИК 2: SHENZHEN CXR (нормальные)
# ─────────────────────────────────────────────
SHENZHEN_BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/"
    "Tuberculosis-Chest-X-ray-Datasets/"
    "Shenzhen-Hospital-CXR-Set/CXR_png/"
)
# ~326 нормальных: CHNCXR_0001_0.png ... (снова _0 = нормальные)
SHENZHEN_NORMAL = [f"CHNCXR_{str(i).zfill(4)}_0.png" for i in range(1, 201)]


def download_shenzhen_normal(dest_dir, max_count):
    """Скачивает нормальные рентгены из Shenzhen Hospital CXR dataset."""
    log("Source 2: Shenzhen Hospital CXR (normal lungs)", "DL")
    ok = 0
    for fname in SHENZHEN_NORMAL:
        if ok >= max_count: break
        dest = Path(dest_dir) / fname
        url  = SHENZHEN_BASE + fname
        if dest.exists(): ok+=1; continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"MedAI/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            with open(dest, "wb") as f: f.write(data)
            ok+=1; time.sleep(CFG["DELAY"])
        except: pass
        if ok % 50 == 0 and ok > 0:
            log(f"  Shenzhen normal: {ok} downloaded", "INFO")
    log(f"Shenzhen: {ok} normal images", "OK")
    return ok


# ─────────────────────────────────────────────
#  ИСТОЧНИК 3: NIH ChestX-ray14 mini-links
# ─────────────────────────────────────────────
# NIH предоставляет прямые ссылки на подмножество (batch_download_zips.txt)
# Мы используем официальную публичную ссылку на NIH images_001.tar.gz
# и забираем оттуда только Pneumonia-помеченные и No Finding-помеченные

NIH_META_URL = (
    "https://nihcc.app.box.com/shared/static/"
    "vfk49d74nhbxq3nqjg0900w3ywvej3n7.gz"
)

# Прямые зеркальные ссылки к NIH (используются если основной недоступен)
NIH_OPEN_MIRRORS = [
    # Sample images из публичного репозитория NIH на GitHub
    "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/",
]


def download_nih_sample(dest_normal, dest_pneumonia, max_each):
    """
    Скачивает публичные образцы NIH chest xray.
    Использует OpenI (Indiana University) как запасной источник.
    """
    log("Source 3: OpenI (Indiana Univ. CXR) pneumonia samples", "DL")
    # OpenI REST API — публичный, без авторизации
    # https://openi.nlm.nih.gov/api/search
    base_api = "https://openi.nlm.nih.gov/api/search"
    ok_pn, ok_nm = 0, 0

    for query, target_dir, label, counter in [
        ("pneumonia", dest_pneumonia, "pneumonia", "ok_pn"),
        ("normal",    dest_normal,    "normal",    "ok_nm"),
    ]:
        n = count_images(target_dir)
        if n >= max_each:
            log(f"  {label}: already {n} images", "OK"); continue

        params = urllib.parse.urlencode({
            "query": query, "it": "x",
            "ctype": "chest", "m": "m,s",
            "n": min(max_each, 100), "skip": 0
        })
        url = f"{base_api}?{params}"
        try:
            req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"MedAI/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
        except Exception as e:
            log(f"  OpenI API failed: {e}", "WARN"); continue

        results = data.get("list", [])
        log(f"  OpenI returned {len(results)} {label} records", "INFO")

        for item in results:
            img_url = item.get("imgLarge") or item.get("imgSmall","")
            if not img_url: continue
            if not img_url.startswith("http"):
                img_url = "https://openi.nlm.nih.gov" + img_url
            fname = f"openi_{label}_{Path(img_url).name}"
            dest  = Path(target_dir) / fname
            if dest.exists():
                if label=="pneumonia": ok_pn+=1
                else: ok_nm+=1
                continue
            try:
                req2 = urllib.request.Request(img_url, headers={"User-Agent":"MedAI/1.0"})
                with urllib.request.urlopen(req2, timeout=20) as r2:
                    data2 = r2.read()
                with open(dest,"wb") as f: f.write(data2)
                if label=="pneumonia": ok_pn+=1
                else: ok_nm+=1
                time.sleep(CFG["DELAY"])
            except: pass

        log(f"  OpenI {label}: {ok_pn if label=='pneumonia' else ok_nm} downloaded", "OK")

    return ok_pn, ok_nm


# ─────────────────────────────────────────────
#  ИСТОЧНИК 4: COVID-19 Chest X-Ray (GitHub, публичный)
# ─────────────────────────────────────────────
COVID_METADATA_URL = (
    "https://raw.githubusercontent.com/ieee8023/"
    "covid-chestxray-dataset/master/metadata.csv"
)
COVID_IMAGE_BASE = (
    "https://raw.githubusercontent.com/ieee8023/"
    "covid-chestxray-dataset/master/images/"
)


def download_covid_pneumonia(dest_dir, max_count):
    """
    Скачивает рентгены пневмонии из COVID-19 Chest X-Ray Dataset.
    Датасет содержит снимки вирусной и бактериальной пневмонии (не только COVID).
    Это публичный GitHub репозиторий.
    """
    log("Source 4: COVID-19 CXR Dataset (GitHub, pneumonia)", "DL")
    cache = CFG["CACHE_DIR"]

    meta_path = os.path.join(cache, "covid_metadata.csv")
    if not _fetch_url(COVID_METADATA_URL, meta_path, "metadata"):
        log("Cannot download COVID dataset metadata", "WARN"); return 0

    ok = 0
    try:
        with open(meta_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if ok >= max_count: break
                finding = row.get("finding","").lower()
                # Берём пневмонию (но не COVID, чтобы избежать редких случаев)
                if not any(x in finding for x in ["pneumonia","bacteria","viral"]):
                    continue
                if "covid" in finding and "pneumonia" not in finding:
                    continue
                fname = row.get("filename","")
                if not fname: continue
                dest = Path(dest_dir) / fname
                if dest.exists(): ok+=1; continue
                url = COVID_IMAGE_BASE + fname
                try:
                    req = urllib.request.Request(url, headers={"User-Agent":"MedAI/1.0"})
                    with urllib.request.urlopen(req, timeout=25) as r:
                        data = r.read()
                    with open(dest,"wb") as f2: f2.write(data)
                    ok+=1; time.sleep(CFG["DELAY"])
                    if ok % 20 == 0: log(f"  COVID pneumonia: {ok}", "INFO")
                except: pass
    except Exception as e:
        log(f"Error reading COVID metadata: {e}", "ERR")

    log(f"COVID dataset: {ok} pneumonia images", "OK")
    return ok


# ─────────────────────────────────────────────
#  ГЛАВНАЯ ФУНКЦИЯ СКАЧИВАНИЯ
# ─────────────────────────────────────────────
def run_download(max_per_class, force=False):
    banner("STEP 1 — DOWNLOAD CHEST DATASET", "─")
    ensure_dirs()

    normal_dir    = os.path.join(CFG["DATA_DIR"], "normal")
    pneumonia_dir = os.path.join(CFG["DATA_DIR"], "pneumonia")

    # ── Нормальные лёгкие ────────────────────
    if force or not os.path.exists(FLAG_NORMAL):
        log(f"Downloading NORMAL lungs (target: {max_per_class})", "DL")
        n1 = download_montgomery_normal(normal_dir, max_per_class//2)
        need = max_per_class - count_images(normal_dir)
        if need > 0:
            n2 = download_shenzhen_normal(normal_dir, need)
        total_nm = count_images(normal_dir)
        if total_nm > 10:
            with open(FLAG_NORMAL,"w") as f:
                json.dump({"count":total_nm,"date":datetime.now().isoformat()},f)
    else:
        log(f"Normal lungs already downloaded ({count_images(normal_dir)}) — skipping", "OK")

    # ── Пневмония ────────────────────────────
    if force or not os.path.exists(FLAG_PNEUMONIA):
        log(f"Downloading PNEUMONIA (target: {max_per_class})", "DL")
        # Сначала COVID репозиторий
        pn1 = download_covid_pneumonia(pneumonia_dir, max_per_class)
        # Затем OpenI если нужно больше
        need = max_per_class - count_images(pneumonia_dir)
        if need > 0:
            pn2, _ = download_nih_sample(normal_dir, pneumonia_dir, need)
        total_pn = count_images(pneumonia_dir)
        if total_pn > 10:
            with open(FLAG_PNEUMONIA,"w") as f:
                json.dump({"count":total_pn,"date":datetime.now().isoformat()},f)
    else:
        log(f"Pneumonia already downloaded ({count_images(pneumonia_dir)}) — skipping", "OK")

    _show_stats()


def _show_stats():
    print()
    banner("DATASET STATISTICS", "─")
    total = 0
    for cls in CFG["CLASSES"]:
        n = count_images(os.path.join(CFG["DATA_DIR"], cls))
        bar = "█" * min(int(n/20),50)
        total += n
        tag = " ← positive" if cls == CFG["POSITIVE_CLASS"] else ""
        print(f"  {cls:12}: {n:6,}  {bar}{tag}")
    print(f"  {'TOTAL':12}: {total:6,}")


# ─────────────────────────────────────────────
#  ОБУЧЕНИЕ
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, smoothing=0.03):
        super().__init__()
        self.gamma=gamma; self.smoothing=smoothing; self.weight=weight
    def forward(self, preds, targets):
        ce = F.cross_entropy(preds,targets,weight=self.weight,
                             label_smoothing=self.smoothing,reduction='none')
        return (((1-torch.exp(-ce))**self.gamma)*ce).mean()


def _compute_class_weights():
    counts = {cls: max(count_images(os.path.join(CFG["DATA_DIR"],cls)),1)
              for cls in CFG["CLASSES"]}
    total = sum(counts.values())
    n = len(counts)
    w = [total/(n*counts[cls]) for cls in CFG["CLASSES"]]
    return torch.tensor(w, dtype=torch.float32)


def run_train():
    try:
        from fastai.vision.all import (
            ImageDataLoaders, vision_learner, aug_transforms,
            Resize, RandomResizedCrop, Normalize, imagenet_stats,
            SaveModelCallback, EarlyStoppingCallback, GradientClip,
            accuracy, valley, slide
        )
        from sklearn.metrics import (precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, average_precision_score,
            balanced_accuracy_score, roc_curve)
    except ImportError as e:
        log(f"Missing dependency: {e}. Run: pip install fastai scikit-learn", "ERR"); return

    banner("STEP 2 — TRAINING CHEST CLASSIFIER", "─")
    set_seed(CFG["SEED"]); gpu_info()

    # Проверяем что данные есть
    for cls in CFG["CLASSES"]:
        n = count_images(os.path.join(CFG["DATA_DIR"],cls))
        if n < 10:
            log(f"Not enough '{cls}' images ({n}). Run with --download-only first.", "ERR")
            return

    # ── DataLoaders ──────────────────────────
    item_tfms  = [Resize(CFG["IMG_SIZE"]+24, method='squish'),
                  RandomResizedCrop(CFG["IMG_SIZE"], min_scale=0.82, ratio=(0.92,1.08))]
    batch_tfms = [*aug_transforms(
                      do_flip=True,
                      flip_vert=CFG["FLIP_VERT"],    # False для рентгена!
                      max_rotate=CFG["MAX_ROTATE"],
                      min_scale=0.90,
                      max_lighting=0.12,
                      max_warp=CFG["MAX_WARP"],
                      p_affine=0.4, p_lighting=0.4),
                  Normalize.from_stats(*imagenet_stats)]

    dls = ImageDataLoaders.from_folder(
        CFG["DATA_DIR"], valid_pct=CFG["VALID_PCT"], seed=CFG["SEED"],
        item_tfms=item_tfms, batch_tfms=batch_tfms,
        num_workers=0, bs=CFG["BATCH_SIZE"])

    log(f"Classes : {list(dls.vocab)}", "OK")
    log(f"Train   : {len(dls.train_ds):,}  |  Valid : {len(dls.valid_ds):,}", "OK")

    # ── Learner ──────────────────────────────
    weights = _compute_class_weights()
    loss_fn = FocalLoss(weight=weights, gamma=2.0, smoothing=CFG["LABEL_SMOOTHING"])

    learn = vision_learner(dls, 'resnet34', metrics=[accuracy],
                           pretrained=True, loss_func=loss_fn,
                           path=Path(CFG["MODELS_DIR"]))
    learn.add_cb(GradientClip(CFG["GRAD_CLIP"]))

    save_cb = SaveModelCallback(monitor='valid_loss', fname='best_chest')
    early   = EarlyStoppingCallback(monitor='valid_loss', patience=CFG["EARLY_STOP_PAT"])

    # ── Phase 1 ──────────────────────────────
    log("Phase 1/3 — Head only", "TRAIN")
    learn.freeze()
    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=30)
        lr1 = min(res.valley, CFG["BASE_LR"])
    except: lr1 = CFG["BASE_LR"]
    learn.fit_one_cycle(CFG["EPOCHS_HEAD"], lr1*10, cbs=[save_cb])

    # ── Phase 2 ──────────────────────────────
    log("Phase 2/3 — Head + last block", "TRAIN")
    learn.freeze_to(-2)
    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=30)
        lr2 = min(res.valley, CFG["BASE_LR"])
    except: lr2 = CFG["BASE_LR"]
    learn.fit_one_cycle(CFG["EPOCHS_UNFREEZE"], slice(lr2/20,lr2), cbs=[save_cb])

    # ── Phase 3 ──────────────────────────────
    log("Phase 3/3 — Full network", "TRAIN")
    learn.unfreeze()
    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=30)
        lr3 = min(res.valley, CFG["BASE_LR"]/5)
    except: lr3 = CFG["BASE_LR"]/10
    learn.fit_one_cycle(CFG["EPOCHS_FULL"], slice(lr3/100,lr3/10),
                        cbs=[save_cb, early])

    best_path = Path(CFG["MODELS_DIR"])/"models"/"best_chest.pth"
    if best_path.exists():
        learn.load("best_chest"); log("Best weights loaded", "OK")

    # ── Evaluation ───────────────────────────
    banner("STEP 3 — EVALUATION", "─")
    preds_tta, targets = learn.tta(n=CFG["TTA_N"], beta=0.35)
    probs    = torch.softmax(preds_tta, dim=1).cpu().numpy()
    targets_ = targets.cpu().numpy()
    vocab    = list(learn.dls.vocab)
    pos_idx  = vocab.index(CFG["POSITIVE_CLASS"])
    scores   = probs[:, pos_idx]

    fpr, tpr, ths = roc_curve(targets_, scores)
    opt_thr  = float(ths[(tpr-fpr).argmax()])
    preds_b  = (scores >= opt_thr).astype(int)

    acc  = float((preds_b==targets_).mean())
    prec = float(precision_score(targets_, preds_b, zero_division=0))
    rec  = float(recall_score(targets_, preds_b, zero_division=0))
    f1   = float(f1_score(targets_, preds_b, zero_division=0))
    auc  = float(roc_auc_score(targets_, scores))
    ap   = float(average_precision_score(targets_, scores))
    bal  = float(balanced_accuracy_score(targets_, preds_b))
    cm   = confusion_matrix(targets_, preds_b)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    spec = float(tn/(tn+fp+1e-8))
    npv  = float(tn/(tn+fn+1e-8))

    print(f"\n  {'─'*52}")
    print(f"  Accuracy              : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy     : {bal:.4f}")
    print(f"  Precision (PPV)       : {prec:.4f}")
    print(f"  Recall (Sensitivity)  : {rec:.4f}  ← ключевая метрика!")
    print(f"  Specificity (TNR)     : {spec:.4f}")
    print(f"  NPV                   : {npv:.4f}")
    print(f"  F1-Score              : {f1:.4f}")
    print(f"  AUC-ROC               : {auc:.4f}")
    print(f"  Avg Precision (AP)    : {ap:.4f}")
    print(f"  Optimal Threshold     : {opt_thr:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  {'─'*52}")

    if rec < 0.85:
        log(f"RECALL={rec:.3f} < 0.85 — слишком много пропущенных пневмоний!", "WARN")

    # ── Сохранение ───────────────────────────
    export_path = os.path.join(CFG["MODELS_DIR"], CFG["EXPORT_NAME"])
    learn.export(export_path)

    metadata = {
        "model_name":     "Lung Disease Detector",
        "model_type":     "lungs",
        "architecture":   "ResNet34 + FocalLoss",
        "classes":        vocab,
        "positive_class": CFG["POSITIVE_CLASS"],
        "img_size":       CFG["IMG_SIZE"],
        "training_date":  datetime.now().isoformat(),
        "performance_metrics": {
            "overall": {
                "accuracy":          round(acc,4),
                "balanced_accuracy": round(bal,4),
                "precision":         round(prec,4),
                "recall":            round(rec,4),
                "specificity":       round(spec,4),
                "npv":               round(npv,4),
                "f1_score":          round(f1,4),
                "auc_roc":           round(auc,4),
                "avg_precision":     round(ap,4),
                "optimal_threshold": round(opt_thr,4),
            },
            "confusion_matrix": {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
        },
        "model_path": export_path,
    }
    meta_path = os.path.join(CFG["MODELS_DIR"], CFG["META_NAME"])
    with open(meta_path,"w",encoding="utf-8") as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)

    log(f"Model  → {export_path}", "OK")
    log(f"Meta   → {meta_path}", "OK")
    log(f"AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}  Spec={spec:.4f}", "OK")
    return metadata


# ─────────────────────────────────────────────
#  ПРЕДСКАЗАНИЕ
# ─────────────────────────────────────────────
def predict(image_path, threshold=None):
    from fastai.vision.all import load_learner, PILImage
    model_path = os.path.join(CFG["MODELS_DIR"], CFG["EXPORT_NAME"])
    if not os.path.exists(model_path):
        log(f"Model not found: {model_path}", "ERR"); return

    meta_path = os.path.join(CFG["MODELS_DIR"], CFG["META_NAME"])
    if threshold is None and os.path.exists(meta_path):
        with open(meta_path) as f: meta = json.load(f)
        threshold = meta.get("performance_metrics",{}).get("overall",{}).get("optimal_threshold",0.5)

    learn = load_learner(model_path)
    img   = PILImage.create(image_path)
    pred_class, pred_idx, probs = learn.predict(img)
    vocab   = list(learn.dls.vocab)
    pos_idx = vocab.index(CFG["POSITIVE_CLASS"])
    conf    = float(probs[pos_idx])
    final   = CFG["POSITIVE_CLASS"] if conf >= threshold else "normal"

    print(f"\n  {'─'*45}")
    print(f"  🫁  CHEST X-RAY PREDICTION")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Confidence : {conf:.1%} (threshold={threshold:.3f})")
    for cls, p in zip(vocab, probs):
        bar = "█"*int(float(p)*30)
        print(f"  {cls:12}: {float(p):.4f}  {bar}")
    status = "⚠️  PNEUMONIA DETECTED" if final=="pneumonia" else "✅ NORMAL LUNGS"
    print(f"  RESULT: {status}")
    print(f"  {'─'*45}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Chest X-ray pneumonia pipeline")
    parser.add_argument("--download-only",  action="store_true")
    parser.add_argument("--train-only",     action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max", type=int, default=CFG["MAX_PER_CLASS"],
                        help=f"Max images per class (default={CFG['MAX_PER_CLASS']})")
    parser.add_argument("--predict",   type=str, help="Path to X-ray image")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    banner("🫁  CHEST X-RAY PIPELINE  (pneumonia vs normal)")

    if args.predict:
        predict(args.predict, args.threshold); return

    ensure_dirs()

    if args.train_only:
        run_train(); return

    run_download(args.max, force=args.force_download)

    if not args.download_only:
        run_train()


if __name__ == "__main__":
    main()