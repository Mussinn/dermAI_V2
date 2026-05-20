"""
skin_pipeline.py — Скачивание данных + Обучение классификатора кожи
════════════════════════════════════════════════════════════════════
Классы: melanoma / normal (nevus)

СПОСОБЫ ПОЛУЧИТЬ ДАННЫЕ (выберите один):

  ── Способ 1: ISIC Archive (РЕКОМЕНДУЕТСЯ) ───────────────────────
     Бесплатная регистрация на https://www.isic-archive.com
     Затем в терминале:
         pip install isic-cli
         isic user login          ← вводите email + пароль один раз
         python skin_pipeline.py  ← дальше всё автоматически

  ── Способ 2: Вручную (HAM10000) ─────────────────────────────────
     1. Скачайте с Kaggle: kaggle.com/datasets/kmader/skin-lesion-analysis
     2. Распакуйте, скопируйте папки:
        skin_data/melanoma/   ← изображения меланомы
        skin_data/normal/     ← нормальная кожа
     3. python skin_pipeline.py --train-only

  ── Способ 3: Kaggle API (автоматически) ─────────────────────────
     pip install opendatasets
     python skin_pipeline.py --source kaggle

Флаги запуска:
    python skin_pipeline.py                    # скачать + обучить
    python skin_pipeline.py --download-only    # только скачать
    python skin_pipeline.py --train-only       # только обучить
    python skin_pipeline.py --force-download   # перескачать
    python skin_pipeline.py --max 1000         # лимит на класс
    python skin_pipeline.py --source kaggle    # источник: kaggle
    python skin_pipeline.py --source isic      # источник: isic (default)
    python skin_pipeline.py --predict img.jpg  # предсказание
"""

import os, sys, json, argparse, warnings, random, time, subprocess, zipfile, csv
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
    "DATA_DIR":          "skin_data",
    "MODELS_DIR":        "skin_models_hq",
    "EXPORT_NAME":       "skin_model_hq.pkl",
    "META_NAME":         "metadata_skin.json",
    "CACHE_DIR":         ".skin_cache",

    "CLASSES":           ["melanoma", "normal"],
    "POSITIVE_CLASS":    "melanoma",
    "MAX_PER_CLASS":     1500,

    # Обучение
    "IMG_SIZE":          224,
    "BATCH_SIZE":        8,
    "EPOCHS_HEAD":       3,
    "EPOCHS_UNFREEZE":   5,
    "EPOCHS_FULL":       8,
    "VALID_PCT":         0.15,
    "SEED":              42,
    "BASE_LR":           1e-3,
    "LABEL_SMOOTHING":   0.05,
    "DROPOUT":           0.35,
    "GRAD_CLIP":         1.0,
    "EARLY_STOP_PAT":    5,
    "TTA_N":             4,
}

FLAG_MELANOMA = os.path.join(CFG["DATA_DIR"], ".done_melanoma")
FLAG_NORMAL   = os.path.join(CFG["DATA_DIR"], ".done_normal")

# ─────────────────────────────────────────────
#  УТИЛИТЫ
# ─────────────────────────────────────────────
def banner(text, char="═"):
    w = 64
    print(f"\n{char*w}\n  {text}\n{char*w}")

def log(msg, tag="INFO"):
    icons = {"INFO":"ℹ️ ","OK":"✅","WARN":"⚠️ ","ERR":"❌","DL":"⬇️ ","TRAIN":"🧠"}
    print(f"  {icons.get(tag,'  ')} {msg}", flush=True)

def ensure_dirs():
    for cls in CFG["CLASSES"]:
        os.makedirs(os.path.join(CFG["DATA_DIR"], cls), exist_ok=True)
    os.makedirs(CFG["MODELS_DIR"], exist_ok=True)
    os.makedirs(CFG["CACHE_DIR"], exist_ok=True)

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
        log("GPU not available — using CPU (slow!)", "WARN")

def _set_flag(flag_path, label, count):
    with open(flag_path, "w") as f:
        json.dump({"label": label, "count": count,
                   "date": datetime.now().isoformat()}, f)

def _show_dataset_stats():
    print()
    banner("DATASET STATISTICS", "─")
    total = 0
    for cls in CFG["CLASSES"]:
        n = count_images(os.path.join(CFG["DATA_DIR"], cls))
        bar = "█" * min(int(n/30), 42)
        total += n
        tag = " ← positive" if cls == CFG["POSITIVE_CLASS"] else ""
        print(f"  {cls:12}: {n:6,}  {bar}{tag}")
    print(f"  {'TOTAL':12}: {total:6,}")


# ════════════════════════════════════════════
#  ИСТОЧНИК 1: ISIC ARCHIVE (через isic-cli)
# ════════════════════════════════════════════
def _run(cmd, **kwargs):
    """subprocess.run с перехватом FileNotFoundError (Windows)."""
    try:
        return subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        class _Fake:
            returncode = 1
            stdout = ""
            stderr = "not found"
        return _Fake()


def _check_isic_cli():
    """Проверяет isic-cli, при необходимости устанавливает и предлагает войти."""
    r = _run(["isic", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        log("isic-cli not found. Installing...", "WARN")
        subprocess.run([sys.executable, "-m", "pip", "install", "isic-cli", "-q"],
                       check=False)
        r = _run(["isic", "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            log("isic-cli not installed. Run: pip install isic-cli", "ERR")
            return False
    log(f"isic-cli: {r.stdout.strip()}", "OK")

    # Проверяем авторизацию — тестовый запрос с лимитом 1
    os.makedirs(os.path.join(CFG["CACHE_DIR"], "_test"), exist_ok=True)
    test = _run(
        ["isic", "image", "download", "--limit", "1", "--search",
         'diagnosis_3:"Melanoma Invasive"', os.path.join(CFG["CACHE_DIR"], "_test")],
        capture_output=True, text=True, timeout=30
    )
    combined = (test.stdout or "") + (test.stderr or "")
    if "403" in combined or "Forbidden" in combined or "unauthorized" in combined.lower():
        print()
        log("ISIC Archive требует авторизацию (бесплатно):", "WARN")
        log("  1. Зарегистрируйтесь: https://login.isic-archive.com/", "WARN")
        log("  2. В терминале выполните: isic user login", "WARN")
        log("  3. Введите email и пароль", "WARN")
        log("  4. Запустите скрипт снова", "WARN")
        print()
        try:
            answer = input("  Хотите войти сейчас? (y/n): ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            result = _run(["isic", "user", "login"])
            if result.returncode == 0:
                log("Logged in successfully!", "OK")
                return True
        return False
    return True


def download_isic(max_per_class, force=False):
    """Скачивает данные через isic-cli (требует бесплатный аккаунт ISIC)."""
    banner("DOWNLOAD via ISIC Archive (isic-cli)", "─")

    if not _check_isic_cli():
        return False

    # Правильный синтаксис поиска ISIC (проверено в isic-cli v12+)
    search_map = {
        "melanoma": 'diagnosis_3:"Melanoma Invasive"',
        "normal":   'diagnosis_3:"Nevus"',
    }
    flag_map = {
        "melanoma": FLAG_MELANOMA,
        "normal":   FLAG_NORMAL,
    }

    for cls in CFG["CLASSES"]:
        flag = flag_map[cls]
        dest = os.path.join(CFG["DATA_DIR"], cls)

        if not force and os.path.exists(flag):
            n = count_images(dest)
            log(f"'{cls}' already downloaded ({n}) — skipping", "OK")
            continue

        os.makedirs(dest, exist_ok=True)
        search = search_map[cls]
        log(f"Downloading '{cls}': isic search='{search}' limit={max_per_class}", "DL")

        cmd = ["isic", "image", "download",
               "--search", search,
               "--limit", str(max_per_class),
               dest]
        log(f"Command: {' '.join(cmd)}", "INFO")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line: print(f"    {line}")
            proc.wait()
        except FileNotFoundError:
            log("isic not found after install. Restart terminal and retry.", "ERR")
            return False
        except Exception as e:
            log(f"isic download error: {e}", "ERR")
            return False

        n = count_images(dest)
        if n > 10:
            _set_flag(flag, cls, n)
            log(f"'{cls}': {n} images downloaded ✓", "OK")
        else:
            log(f"'{cls}': only {n} images. Check login and try again.", "WARN")

    return True


# ════════════════════════════════════════════
#  ИСТОЧНИК 2: KAGGLE (через opendatasets)
# ════════════════════════════════════════════
def download_kaggle(max_per_class, force=False):
    """
    Скачивает HAM10000 через opendatasets.
    При первом запуске спросит Kaggle username и API key.
    Получить ключ: https://www.kaggle.com/settings → API → Create New Token
    """
    banner("DOWNLOAD via Kaggle (HAM10000)", "─")

    # Проверяем флаги
    all_done = all(
        (not force and os.path.exists(f) and count_images(os.path.join(CFG["DATA_DIR"], c)) > 10)
        for c, f in [("melanoma", FLAG_MELANOMA), ("normal", FLAG_NORMAL)]
    )
    if all_done:
        log("All classes already downloaded — skipping", "OK")
        return True

    try:
        import opendatasets as od
    except ImportError:
        log("Installing opendatasets...", "WARN")
        subprocess.run([sys.executable, "-m", "pip", "install", "opendatasets", "-q"],
                       check=True)
        import opendatasets as od

    cache = CFG["CACHE_DIR"]
    log("Downloading HAM10000 from Kaggle...", "DL")
    log("You will be asked for Kaggle username and API key.", "INFO")
    log("Get your key: https://www.kaggle.com/settings → API → Create New Token", "INFO")

    try:
        od.download(
            "https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection",
            data_dir=cache
        )
    except Exception as e:
        log(f"Kaggle download failed: {e}", "ERR")
        return False

    return _sort_ham10000(cache, max_per_class)


def _sort_ham10000(cache_dir, max_per_class):
    """Сортирует HAM10000 по классам используя CSV метаданные."""
    mel_dir  = os.path.join(CFG["DATA_DIR"], "melanoma")
    norm_dir = os.path.join(CFG["DATA_DIR"], "normal")
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    # Ищем CSV с метаданными
    meta_csv = None
    for root, _, files in os.walk(cache_dir):
        for f in files:
            if 'metadata' in f.lower() and f.endswith('.csv'):
                meta_csv = os.path.join(root, f)
                break
        if meta_csv: break

    if not meta_csv:
        log("HAM10000 metadata CSV not found in cache", "ERR")
        return False

    log(f"Using metadata: {meta_csv}", "INFO")

    # dx коды: mel=меланома, остальные=нормальные
    # nv=nevus, bkl=себорейный кератоз, df=дерматофиброма, vasc=сосудистые
    NORMAL_DX = {"nv", "bkl", "df", "vasc", "akiec"}

    img_class = {}
    with open(meta_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            dx = row.get("dx", "").lower().strip()
            img_id = row.get("image_id", "").strip()
            if dx == "mel":
                img_class[img_id] = "melanoma"
            elif dx in NORMAL_DX:
                img_class[img_id] = "normal"

    log(f"Found {sum(1 for v in img_class.values() if v=='melanoma')} melanoma, "
        f"{sum(1 for v in img_class.values() if v=='normal')} normal in metadata", "INFO")

    mel_count = count_images(mel_dir)
    norm_count = count_images(norm_dir)
    copied = 0

    # Ищем и копируем изображения
    for root, _, files in os.walk(cache_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
            img_id = Path(fname).stem
            cls = img_class.get(img_id)
            if not cls: continue

            if cls == "melanoma" and mel_count < max_per_class:
                dst = os.path.join(mel_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(root, fname), dst)
                mel_count += 1; copied += 1
            elif cls == "normal" and norm_count < max_per_class:
                dst = os.path.join(norm_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(root, fname), dst)
                norm_count += 1; copied += 1

    log(f"Sorted: melanoma={mel_count}, normal={norm_count} ({copied} copied)", "OK")

    if mel_count > 10: _set_flag(FLAG_MELANOMA, "melanoma", mel_count)
    if norm_count > 10: _set_flag(FLAG_NORMAL, "normal", norm_count)
    return mel_count > 10 and norm_count > 10


# ════════════════════════════════════════════
#  ИСТОЧНИК 3: РУЧНОЙ ZIP
# ════════════════════════════════════════════
def try_manual_zip(max_per_class):
    """
    Если пользователь вручную скачал HAM10000.zip или skin_data.zip —
    распаковываем и сортируем.
    """
    zip_candidates = list(Path(".").glob("*.zip")) + list(Path(CFG["CACHE_DIR"]).glob("*.zip"))

    for zip_path in zip_candidates:
        if any(kw in zip_path.name.lower() for kw in ["ham", "skin", "isic", "melanoma"]):
            log(f"Found ZIP: {zip_path}", "INFO")
            extract_dir = os.path.join(CFG["CACHE_DIR"], "zip_extracted")
            os.makedirs(extract_dir, exist_ok=True)
            log(f"Extracting {zip_path}...", "DL")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            return _sort_ham10000(extract_dir, max_per_class)

    return False


# ════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ СКАЧИВАНИЯ
# ════════════════════════════════════════════
def run_download(max_per_class, force=False, source="isic"):
    banner("STEP 1 — DOWNLOAD SKIN DATASET", "─")
    ensure_dirs()

    # Проверяем флаги
    mel_done  = not force and os.path.exists(FLAG_MELANOMA) and count_images(os.path.join(CFG["DATA_DIR"],"melanoma")) > 10
    norm_done = not force and os.path.exists(FLAG_NORMAL)   and count_images(os.path.join(CFG["DATA_DIR"],"normal"))   > 10

    if mel_done and norm_done:
        log("Both classes already downloaded — skipping download", "OK")
        _show_dataset_stats()
        return True

    # Попробуем ZIP файл в текущей папке (наивысший приоритет если есть)
    if try_manual_zip(max_per_class):
        log("Data loaded from local ZIP file", "OK")
        _show_dataset_stats()
        return True

    if source == "kaggle":
        ok = download_kaggle(max_per_class, force)
    else:  # isic (default)
        ok = download_isic(max_per_class, force)
        if not ok:
            log("ISIC failed. Trying Kaggle as fallback...", "WARN")
            ok = download_kaggle(max_per_class, force)

    _show_dataset_stats()
    return ok


# ─────────────────────────────────────────────
#  МОДЕЛЬ
# ─────────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),(1,1)
        ).pow(1./self.p)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, smoothing=0.05):
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


# ─────────────────────────────────────────────
#  ОБУЧЕНИЕ
# ─────────────────────────────────────────────
def run_train():
    try:
        from fastai.vision.all import (
            ImageDataLoaders, vision_learner, aug_transforms,
            Resize, RandomResizedCrop, Normalize, imagenet_stats,
            SaveModelCallback, EarlyStoppingCallback, GradientClip,
            accuracy, valley
        )
        from sklearn.metrics import (precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, average_precision_score,
            balanced_accuracy_score, roc_curve)
    except ImportError as e:
        log(f"Missing: {e}. Run: pip install fastai scikit-learn", "ERR")
        return

    banner("STEP 2 — TRAINING SKIN CLASSIFIER", "─")
    set_seed(CFG["SEED"]); gpu_info()

    # ── Проверка данных ───────────────────────
    for cls in CFG["CLASSES"]:
        n = count_images(os.path.join(CFG["DATA_DIR"], cls))
        log(f"  {cls}: {n} images", "INFO")
        if n < 10:
            log(f"Not enough '{cls}' images ({n}). Download data first:", "ERR")
            log("  Option 1: pip install isic-cli && isic user login", "ERR")
            log("            python skin_pipeline.py --download-only", "ERR")
            log("  Option 2: python skin_pipeline.py --source kaggle --download-only", "ERR")
            log("  Option 3: Place HAM10000.zip next to this script and rerun", "ERR")
            return

    # ── DataLoaders ───────────────────────────
    item_tfms = [
        Resize(CFG["IMG_SIZE"]+32, method='squish'),
        RandomResizedCrop(CFG["IMG_SIZE"], min_scale=0.75, ratio=(0.9,1.1))
    ]
    batch_tfms = [
        *aug_transforms(do_flip=True, flip_vert=True,
                        max_rotate=20, min_scale=0.85,
                        max_lighting=0.2, max_warp=0.1,
                        p_affine=0.5, p_lighting=0.5),
        Normalize.from_stats(*imagenet_stats)
    ]

    dls = ImageDataLoaders.from_folder(
        CFG["DATA_DIR"], valid_pct=CFG["VALID_PCT"], seed=CFG["SEED"],
        item_tfms=item_tfms, batch_tfms=batch_tfms,
        num_workers=0, bs=CFG["BATCH_SIZE"]
    )
    log(f"Classes : {list(dls.vocab)}", "OK")
    log(f"Train   : {len(dls.train_ds):,}  |  Valid : {len(dls.valid_ds):,}", "OK")

    # ── Learner ───────────────────────────────
    weights = _compute_class_weights()
    loss_fn = FocalLoss(weight=weights, gamma=2.0, smoothing=CFG["LABEL_SMOOTHING"])

    learn = vision_learner(
        dls, 'resnet34', metrics=[accuracy],
        pretrained=True, loss_func=loss_fn,
        path=Path(CFG["MODELS_DIR"])
    )
    learn.add_cb(GradientClip(CFG["GRAD_CLIP"]))

    save_cb = SaveModelCallback(monitor='valid_loss', fname='best_skin')
    early   = EarlyStoppingCallback(monitor='valid_loss', patience=CFG["EARLY_STOP_PAT"])

    # ── Phase 1: head only ────────────────────
    log("Phase 1/3 — Head only", "TRAIN")
    learn.freeze()
    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=30)
        lr1 = min(res.valley, CFG["BASE_LR"])
    except: lr1 = CFG["BASE_LR"]
    log(f"LR = {lr1:.2e}", "INFO")
    learn.fit_one_cycle(CFG["EPOCHS_HEAD"], lr1*10, cbs=[save_cb])

    # ── Phase 2: last 2 blocks ────────────────
    log("Phase 2/3 — Head + last block", "TRAIN")
    learn.freeze_to(-2)
    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=30)
        lr2 = min(res.valley, CFG["BASE_LR"])
    except: lr2 = CFG["BASE_LR"]
    learn.fit_one_cycle(CFG["EPOCHS_UNFREEZE"], slice(lr2/20, lr2), cbs=[save_cb])

    # ── Phase 3: full network ─────────────────
    log("Phase 3/3 — Full network", "TRAIN")
    learn.unfreeze()
    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=30)
        lr3 = min(res.valley, CFG["BASE_LR"]/5)
    except: lr3 = CFG["BASE_LR"]/10
    learn.fit_one_cycle(CFG["EPOCHS_FULL"], slice(lr3/100, lr3/10),
                        cbs=[save_cb, early])

    best = Path(CFG["MODELS_DIR"]) / "models" / "best_skin.pth"
    if best.exists():
        learn.load("best_skin"); log("Best weights loaded", "OK")

    # ── Evaluation ────────────────────────────
    banner("STEP 3 — EVALUATION", "─")
    preds_tta, targets = learn.tta(n=CFG["TTA_N"], beta=0.35)
    probs    = torch.softmax(preds_tta, dim=1).cpu().numpy()
    targets_ = targets.cpu().numpy()
    vocab    = list(learn.dls.vocab)
    pos_idx  = vocab.index(CFG["POSITIVE_CLASS"])
    scores   = probs[:, pos_idx]

    fpr, tpr, ths = roc_curve(targets_, scores)
    opt_thr = float(ths[(tpr-fpr).argmax()])
    preds_b = (scores >= opt_thr).astype(int)

    acc  = float((preds_b==targets_).mean())
    prec = float(precision_score(targets_, preds_b, zero_division=0))
    rec  = float(recall_score(targets_, preds_b, zero_division=0))
    f1   = float(f1_score(targets_, preds_b, zero_division=0))
    auc  = float(roc_auc_score(targets_, scores))
    ap   = float(average_precision_score(targets_, scores))
    bal  = float(balanced_accuracy_score(targets_, preds_b))
    cm   = confusion_matrix(targets_, preds_b)
    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)

    print(f"\n  {'─'*54}")
    print(f"  Accuracy              : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy     : {bal:.4f}")
    print(f"  Precision (PPV)       : {prec:.4f}")
    print(f"  Recall (Sensitivity)  : {rec:.4f}")
    print(f"  F1-Score              : {f1:.4f}")
    print(f"  AUC-ROC               : {auc:.4f}")
    print(f"  Avg Precision (AP)    : {ap:.4f}")
    print(f"  Optimal Threshold     : {opt_thr:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  {'─'*54}")

    # ── Сохранение ────────────────────────────
    export_path = os.path.join(CFG["MODELS_DIR"], CFG["EXPORT_NAME"])
    learn.export(export_path)

    metadata = {
        "model_name":      "Skin Cancer Detector",
        "model_type":      "skin",
        "architecture":    "ResNet34 + GeM + FocalLoss",
        "classes":         vocab,
        "positive_class":  CFG["POSITIVE_CLASS"],
        "img_size":        CFG["IMG_SIZE"],
        "training_date":   datetime.now().isoformat(),
        "performance_metrics": {
            "overall": {
                "accuracy":          round(acc,4),
                "balanced_accuracy": round(bal,4),
                "precision":         round(prec,4),
                "recall":            round(rec,4),
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
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log(f"Model  → {export_path}", "OK")
    log(f"Meta   → {meta_path}", "OK")
    log(f"AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}", "OK")
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
        threshold = meta["performance_metrics"]["overall"].get("optimal_threshold", 0.5)

    learn = load_learner(model_path)
    img   = PILImage.create(image_path)
    _, pred_idx, probs = learn.predict(img)
    vocab   = list(learn.dls.vocab)
    pos_idx = vocab.index(CFG["POSITIVE_CLASS"])
    conf    = float(probs[pos_idx])
    final   = CFG["POSITIVE_CLASS"] if conf >= threshold else "normal"

    print(f"\n  {'─'*47}")
    print(f"  🩺  SKIN PREDICTION")
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Threshold : {threshold:.3f}")
    for cls, p in zip(vocab, probs):
        bar = "█"*int(float(p)*30)
        print(f"  {cls:12}: {float(p):.4f}  {bar}")
    status = "⚠️  MELANOMA DETECTED" if final=="melanoma" else "✅ NORMAL SKIN"
    print(f"  RESULT: {status}")
    print(f"  {'─'*47}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Skin cancer pipeline (melanoma vs normal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  Option A (ISIC — recommended):
    pip install isic-cli
    isic user login             ← free account at isic-archive.com
    python skin_pipeline.py

  Option B (Kaggle):
    python skin_pipeline.py --source kaggle

  Option C (manual ZIP):
    Place HAM10000.zip next to this script
    python skin_pipeline.py --download-only
        """
    )
    parser.add_argument("--download-only",  action="store_true")
    parser.add_argument("--train-only",     action="store_true")
    parser.add_argument("--force-download", action="store_true",
                        help="Ignore download flags, re-download")
    parser.add_argument("--max", type=int, default=CFG["MAX_PER_CLASS"],
                        help=f"Max images per class (default={CFG['MAX_PER_CLASS']})")
    parser.add_argument("--source", choices=["isic","kaggle"], default="isic",
                        help="Data source (default: isic)")
    parser.add_argument("--predict",   type=str, help="Path to image")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    banner("🩺  SKIN CANCER PIPELINE  (melanoma vs normal)")

    if args.predict:
        predict(args.predict, args.threshold); return

    ensure_dirs()

    if args.train_only:
        run_train(); return

    ok = run_download(args.max, force=args.force_download, source=args.source)

    if not args.download_only:
        if ok or any(count_images(os.path.join(CFG["DATA_DIR"],c)) > 10
                     for c in CFG["CLASSES"]):
            run_train()
        else:
            log("No images found. Download data first.", "ERR")


if __name__ == "__main__":
    main()
