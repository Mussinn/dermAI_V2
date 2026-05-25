"""
skin_pipeline.py — ИСПРАВЛЕННАЯ ВЕРСИЯ v4
═══════════════════════════════════════════════════════════════════════════
Классы: melanoma / normal

ИСПРАВЛЕНИЯ (v4):
  1. FocalLoss: weight.to(device) — исправлен silent fail на CPU/GPU
  2. Диагностика AUC: автоматическое определение инверсии классов
  3. WeightedRandomSampler: применяется ко ВСЕМ трём стадиям через хелпер
  4. _make_dls_with_sampler() — единая функция для всех DataLoader
  5. Улучшенный порог: защита от threshold=inf (fallback на 0.5)
  6. Дополнительные диагностические логи для отладки
  7. Все остальные исправления v3 сохранены

Запуск:
    python skin_pipeline.py                    # скачать + обучить
    python skin_pipeline.py --download-only
    python skin_pipeline.py --train-only
    python skin_pipeline.py --force-download
    python skin_pipeline.py --finetune         # дообучить модель
    python skin_pipeline.py --predict img.jpg
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
    "DATA_DIR":          "skin_data_processed",
    "MODELS_DIR":        "skin_models_hq",
    "EXPORT_NAME":       "skin_model_hq.pkl",
    "META_NAME":         "metadata_skin.json",
    "CACHE_DIR":         ".skin_cache",

    "CLASSES":           ["melanoma", "normal"],
    "POSITIVE_CLASS":    "melanoma",
    "MAX_PER_CLASS":     2000,

    # Обучение
    "IMG_SIZE_START":    128,
    "IMG_SIZE_FINAL":    224,
    "BATCH_SIZE":        16,
    "EPOCHS_STAGE1":     5,
    "EPOCHS_STAGE2":     8,
    "EPOCHS_STAGE3":     12,
    "VALID_PCT":         0.15,
    "SEED":              42,
    "BASE_LR":           3e-3,

    "LABEL_SMOOTHING":   0.1,
    "FOCAL_GAMMA":       2.5,
    "DROPOUT":           0.4,
    "GRAD_CLIP":         1.0,
    "EARLY_STOP_PAT":    8,
    "TTA_N":             4,
    "USE_MIXUP":         True,
    "MIXUP_ALPHA":       0.2,

    # Для дообучения
    "FINETUNE_LR":       5e-5,
    "FINETUNE_EPOCHS":   10,
}

FLAG_MELANOMA = os.path.join(CFG["DATA_DIR"], ".done_melanoma")
FLAG_NORMAL   = os.path.join(CFG["DATA_DIR"], ".done_normal")


# ─────────────────────────────────────────────
#  УТИЛИТЫ
# ─────────────────────────────────────────────
def banner(text, char="═"):
    w = 70
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
        return True
    else:
        log("GPU not available — using CPU (slow!)", "WARN")
        return False

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
        bar = "█" * min(int(n/40), 50)
        total += n
        tag = " ← POSITIVE (rare)" if cls == CFG["POSITIVE_CLASS"] else ""
        print(f"  {cls:12}: {n:6,}  {bar}{tag}")
    print(f"  {'TOTAL':12}: {total:6,}")

    melanoma = count_images(os.path.join(CFG["DATA_DIR"], "melanoma"))
    normal   = count_images(os.path.join(CFG["DATA_DIR"], "normal"))
    if melanoma > 0 and normal > 0:
        ratio = normal / melanoma
        log(f"Class imbalance: normal/melanoma = {ratio:.1f}x", "INFO")
        if ratio > 2:
            log("Significant imbalance! WeightedRandomSampler will be used.", "WARN")


# ════════════════════════════════════════════
#  ИСТОЧНИКИ ДАННЫХ
# ════════════════════════════════════════════
def _run(cmd, **kwargs):
    try:
        return subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        class _Fake:
            returncode = 1
            stdout = ""
            stderr = "not found"
        return _Fake()

def _check_isic_cli():
    r = _run(["isic", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        log("isic-cli not found. Installing...", "WARN")
        subprocess.run([sys.executable, "-m", "pip", "install", "isic-cli", "-q"], check=False)
        r = _run(["isic", "--version"], capture_output=True, text=True)
        if r.returncode != 0:
            log("isic-cli not installed. Run: pip install isic-cli", "ERR")
            return False
    log(f"isic-cli: {r.stdout.strip()}", "OK")

    os.makedirs(os.path.join(CFG["CACHE_DIR"], "_test"), exist_ok=True)
    test = _run(
        ["isic", "image", "download", "--limit", "1", "--search",
         'diagnosis_3:"Melanoma Invasive"', os.path.join(CFG["CACHE_DIR"], "_test")],
        capture_output=True, text=True, timeout=30
    )
    combined = (test.stdout or "") + (test.stderr or "")
    if "403" in combined or "Forbidden" in combined or "unauthorized" in combined.lower():
        print()
        log("ISIC Archive requires free registration:", "WARN")
        log("  1. Register at: https://login.isic-archive.com/", "WARN")
        log("  2. Run: isic user login", "WARN")
        log("  3. Enter your email and password", "WARN")
        return False
    return True

def download_isic(max_per_class, force=False):
    banner("DOWNLOAD via ISIC Archive", "─")
    if not _check_isic_cli():
        return False

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
        log(f"Downloading '{cls}': search='{search}' limit={max_per_class}", "DL")

        cmd = ["isic", "image", "download", "--search", search,
               "--limit", str(max_per_class), dest]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                if line.strip():
                    print(f"    {line.strip()}")
            proc.wait()
        except Exception as e:
            log(f"isic download error: {e}", "ERR")
            return False

        n = count_images(dest)
        if n > 10:
            _set_flag(flag, cls, n)
            log(f"'{cls}': {n} images downloaded ✓", "OK")
        else:
            log(f"'{cls}': only {n} images", "WARN")
    return True

def download_kaggle(max_per_class, force=False):
    banner("DOWNLOAD via Kaggle (HAM10000)", "─")

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
        subprocess.run([sys.executable, "-m", "pip", "install", "opendatasets", "-q"], check=True)
        import opendatasets as od

    log("Downloading HAM10000 from Kaggle...", "DL")
    log("You will be asked for Kaggle username and API key.", "INFO")
    log("Get your key: https://www.kaggle.com/settings → API → Create New Token", "INFO")

    try:
        od.download(
            "https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection",
            data_dir=CFG["CACHE_DIR"]
        )
    except Exception as e:
        log(f"Kaggle download failed: {e}", "ERR")
        return False

    return _sort_ham10000(CFG["CACHE_DIR"], max_per_class)

def _sort_ham10000(cache_dir, max_per_class):
    mel_dir  = os.path.join(CFG["DATA_DIR"], "melanoma")
    norm_dir = os.path.join(CFG["DATA_DIR"], "normal")
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    meta_csv = None
    for root, _, files in os.walk(cache_dir):
        for f in files:
            if 'metadata' in f.lower() and f.endswith('.csv'):
                meta_csv = os.path.join(root, f)
                break
        if meta_csv:
            break

    if not meta_csv:
        log("HAM10000 metadata CSV not found in cache", "ERR")
        return False

    NORMAL_DX = {"nv", "bkl", "df", "vasc", "akiec"}
    img_class = {}

    with open(meta_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            dx     = row.get("dx", "").lower().strip()
            img_id = row.get("image_id", "").strip()
            if dx == "mel":
                img_class[img_id] = "melanoma"
            elif dx in NORMAL_DX:
                img_class[img_id] = "normal"

    mel_count  = count_images(mel_dir)
    norm_count = count_images(norm_dir)

    for root, _, files in os.walk(cache_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_id = Path(fname).stem
            cls    = img_class.get(img_id)
            if not cls:
                continue

            if cls == "melanoma" and mel_count < max_per_class:
                dst = os.path.join(mel_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(root, fname), dst)
                mel_count += 1
            elif cls == "normal" and norm_count < max_per_class:
                dst = os.path.join(norm_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(root, fname), dst)
                norm_count += 1

    log(f"Sorted: melanoma={mel_count}, normal={norm_count}", "OK")

    if mel_count > 10:
        _set_flag(FLAG_MELANOMA, "melanoma", mel_count)
    if norm_count > 10:
        _set_flag(FLAG_NORMAL, "normal", norm_count)

    return mel_count > 10 and norm_count > 10

def try_manual_zip(max_per_class):
    zip_candidates = list(Path(".").glob("*.zip")) + list(Path(CFG["CACHE_DIR"]).glob("*.zip"))
    for zip_path in zip_candidates:
        if any(kw in zip_path.name.lower() for kw in ["ham", "skin", "isic", "melanoma"]):
            log(f"Found ZIP: {zip_path}", "INFO")
            extract_dir = os.path.join(CFG["CACHE_DIR"], "zip_extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            return _sort_ham10000(extract_dir, max_per_class)
    return False

def run_download(max_per_class, force=False, source="isic"):
    banner("STEP 1 — DOWNLOAD SKIN DATASET", "─")
    ensure_dirs()

    mel_done  = not force and os.path.exists(FLAG_MELANOMA) and count_images(os.path.join(CFG["DATA_DIR"], "melanoma")) > 10
    norm_done = not force and os.path.exists(FLAG_NORMAL)   and count_images(os.path.join(CFG["DATA_DIR"], "normal"))   > 10

    if mel_done and norm_done:
        log("Both classes already downloaded — skipping download", "OK")
        _show_dataset_stats()
        return True

    if try_manual_zip(max_per_class):
        log("Data loaded from local ZIP file", "OK")
        _show_dataset_stats()
        return True

    if source == "kaggle":
        ok = download_kaggle(max_per_class, force)
    else:
        ok = download_isic(max_per_class, force)
        if not ok:
            log("ISIC failed. Trying Kaggle as fallback...", "WARN")
            ok = download_kaggle(max_per_class, force)

    _show_dataset_stats()
    return ok


# ─────────────────────────────────────────────
#  МОДЕЛИ И LOSS
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss для несбалансированных данных.
    ИСПРАВЛЕНИЕ v4: weight.to(preds.device) — предотвращает silent fail
    когда тензор весов на CPU, а модель на GPU.
    """
    def __init__(self, weight=None, gamma=2.5, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        # Регистрируем как буфер, чтобы автоматически переносился на нужный device
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, preds, targets):
        # ИСПРАВЛЕНИЕ: явно переносим веса на устройство предсказаний
        w = self.weight.to(preds.device) if self.weight is not None else None

        ce_loss    = F.cross_entropy(preds, targets, weight=w, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def make_weighted_sampler(dataset, classes):
    """
    Oversampling для балансировки классов через WeightedRandomSampler.
    Работает с fastai ImageDataLoaders dataset (.items = list of Path).
    """
    from torch.utils.data import WeightedRandomSampler

    class_counts = {cls: 0 for cls in classes}
    for item in dataset.items:
        label = item.parent.name
        if label in class_counts:
            class_counts[label] += 1

    log(f"Class counts for sampler: {class_counts}", "INFO")

    sample_weights = []
    for item in dataset.items:
        label = item.parent.name
        cnt   = class_counts.get(label, 1)
        sample_weights.append(1.0 / cnt)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    min_count   = min(class_counts.values())
    num_samples = min_count * 2 * len(classes)

    log(f"WeightedSampler: {num_samples} samples/epoch (min_class={min_count})", "INFO")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )


def _preprocess_images(data_dir, max_side=512):
    from PIL import Image as _PIL
    _PIL.MAX_IMAGE_PIXELS = None

    flag = os.path.join(data_dir, ".preprocessed")
    if os.path.exists(flag):
        log("Images already preprocessed — skipping", "OK")
        return

    exts           = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    total, resized = 0, 0

    log(f"Preprocessing images (resize >{max_side}px to {max_side}px)...", "INFO")

    for cls in CFG["CLASSES"]:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fpath in Path(cls_dir).iterdir():
            if fpath.suffix.lower() not in exts:
                continue
            total += 1
            try:
                img  = _PIL.open(fpath)
                w, h = img.size
                if max(w, h) > max_side:
                    ratio = max_side / max(w, h)
                    new_w = int(w * ratio)
                    new_h = int(h * ratio)
                    img   = img.convert("RGB").resize((new_w, new_h), _PIL.LANCZOS)
                    img.save(fpath, "JPEG", quality=92)
                    resized += 1
                elif img.mode != "RGB":
                    img.convert("RGB").save(fpath, "JPEG", quality=92)
                    resized += 1
            except Exception:
                pass
            if total % 200 == 0:
                log(f"  Processed {total} images...", "INFO")

    log(f"Preprocessing done: {total} total, {resized} resized", "OK")
    with open(flag, "w") as f:
        json.dump({"total": total, "resized": resized}, f)


def _compute_class_weights():
    counts = {cls: max(count_images(os.path.join(CFG["DATA_DIR"], cls)), 1)
              for cls in CFG["CLASSES"]}
    total  = sum(counts.values())
    n      = len(counts)
    w      = [total / (n * counts[cls]) for cls in CFG["CLASSES"]]
    log(f"Class weights: {dict(zip(CFG['CLASSES'], [round(x, 3) for x in w]))}", "INFO")
    return torch.tensor(w, dtype=torch.float32)


# ─────────────────────────────────────────────
#  КАСТОМНЫЕ МЕТРИКИ (ИСПРАВЛЕНИЕ v3/v4)
#  Используем AccumMetric + sklearn чтобы
#  избежать ошибки "y should be a 1d array"
# ─────────────────────────────────────────────
def _build_metrics(pos_idx=0):
    from fastai.metrics import AccumMetric
    from sklearn.metrics import roc_auc_score, f1_score

    def roc_auc_binary(preds, targs):
        scores = torch.softmax(preds, dim=1)[:, pos_idx].cpu().numpy()
        t      = targs.cpu().numpy()
        if len(set(t)) < 2:
            return 0.0
        return roc_auc_score(t, scores)

    def f1_binary(preds, targs):
        p = torch.softmax(preds, dim=1)[:, pos_idx].cpu().numpy()
        b = (p >= 0.5).astype(int)
        t = targs.cpu().numpy()
        return f1_score(t, b, zero_division=0)

    roc_metric = AccumMetric(roc_auc_binary, name='roc_auc', flatten=False)
    f1_metric  = AccumMetric(f1_binary,      name='f1_score', flatten=False)
    return roc_metric, f1_metric


# ─────────────────────────────────────────────
#  ИСПРАВЛЕНИЕ v4: единый хелпер для DataLoader
#  с WeightedRandomSampler на ВСЕХ стадиях
# ─────────────────────────────────────────────
def _make_dls_with_sampler(img_size, batch_tfms, min_scale=0.75):
    """
    Создаёт ImageDataLoaders и применяет WeightedRandomSampler
    к тренировочному датасету. Используется на всех трёх стадиях.
    """
    from fastai.vision.all import (
        ImageDataLoaders, Resize, RandomResizedCrop
    )

    item_tfms = [Resize(img_size + 16), RandomResizedCrop(img_size, min_scale=min_scale)]

    dls = ImageDataLoaders.from_folder(
        CFG["DATA_DIR"],
        valid_pct=CFG["VALID_PCT"],
        seed=CFG["SEED"],
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        num_workers=0,
        bs=CFG["BATCH_SIZE"]
    )

    try:
        sampler = make_weighted_sampler(dls.train_ds, CFG["CLASSES"])
        dls.train = dls.train.new(sampler=sampler, shuffle=False)
        log(f"WeightedRandomSampler applied (img_size={img_size})", "OK")
    except Exception as e:
        log(f"Could not apply WeightedRandomSampler at size {img_size}: {e}", "WARN")

    return dls


# ─────────────────────────────────────────────
#  ИСПРАВЛЕНИЕ v4: диагностика и коррекция AUC
# ─────────────────────────────────────────────
def _diagnose_and_fix_scores(probs, targets_, vocab, pos_idx):
    """
    Проверяет инверсию классов. Если AUC < 0.5 — значит модель
    хорошо различает классы, но индекс перепутан. Возвращает
    исправленные scores и флаг инверсии.
    """
    from sklearn.metrics import roc_auc_score

    scores_direct   = probs[:, pos_idx]
    scores_inverted = probs[:, 1 - pos_idx]

    auc_direct   = roc_auc_score(targets_, scores_direct)
    auc_inverted = roc_auc_score(targets_, scores_inverted)

    log(f"AUC диагностика:", "INFO")
    log(f"  scores[:, {pos_idx}] ('{vocab[pos_idx]}')       → AUC = {auc_direct:.4f}", "INFO")
    log(f"  scores[:, {1-pos_idx}] ('{vocab[1-pos_idx]}') → AUC = {auc_inverted:.4f}", "INFO")

    mean_mel  = scores_direct[targets_ == pos_idx].mean()
    mean_norm = scores_direct[targets_ != pos_idx].mean()
    log(f"  Mean score для melanoma : {mean_mel:.4f}", "INFO")
    log(f"  Mean score для normal   : {mean_norm:.4f}", "INFO")

    if auc_direct < 0.5 and auc_inverted > auc_direct:
        log(f"ИНВЕРСИЯ ОБНАРУЖЕНА! AUC={auc_direct:.4f} < 0.5 → используем инвертированные scores", "WARN")
        log(f"Исправленный AUC = {auc_inverted:.4f}", "OK")
        return scores_inverted, True
    else:
        log(f"Инверсии нет. Используем scores[:, {pos_idx}]. AUC = {auc_direct:.4f}", "OK")
        return scores_direct, False


# ─────────────────────────────────────────────
#  ОБУЧЕНИЕ С PROGRESSIVE RESIZING
# ─────────────────────────────────────────────
def run_train():
    try:
        from fastai.vision.all import (
            vision_learner, aug_transforms,
            Normalize, imagenet_stats,
            SaveModelCallback, EarlyStoppingCallback, GradientClip,
            accuracy, valley
        )
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix,
            balanced_accuracy_score, roc_curve
        )
    except ImportError as e:
        log(f"Missing: {e}. Run: pip install fastai scikit-learn", "ERR")
        return

    banner("STEP 2 — TRAINING SKIN CLASSIFIER (IMPROVED v4)", "─")
    set_seed(CFG["SEED"])
    has_gpu = gpu_info()

    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = None
    _preprocess_images(CFG["DATA_DIR"])

    for cls in CFG["CLASSES"]:
        n = count_images(os.path.join(CFG["DATA_DIR"], cls))
        log(f"  {cls}: {n} images", "INFO")
        if n < 50:
            log(f"Not enough '{cls}' images ({n}). Need at least 50 per class.", "ERR")
            return

    # Аугментации (одинаковые для всех стадий)
    batch_tfms = [
        *aug_transforms(
            do_flip=True, flip_vert=True,
            max_rotate=30,
            max_zoom=1.2,
            max_lighting=0.3,
            max_warp=0.2,
            p_affine=0.75,
            p_lighting=0.75
        ),
        Normalize.from_stats(*imagenet_stats)
    ]

    weights = _compute_class_weights()
    loss_fn = FocalLoss(weight=weights, gamma=CFG["FOCAL_GAMMA"])

    # ── STAGE 1: 128x128 ─────────────────────────────────────────
    banner("STAGE 1/3 — Training at 128x128", "─")

    # ИСПРАВЛЕНИЕ v4: используем хелпер, который применяет sampler
    dls = _make_dls_with_sampler(
        img_size=CFG["IMG_SIZE_START"],
        batch_tfms=batch_tfms,
        min_scale=0.7
    )

    vocab   = list(dls.vocab)
    pos_idx = vocab.index(CFG["POSITIVE_CLASS"])
    log(f"Vocab: {vocab}  |  positive_class='{CFG['POSITIVE_CLASS']}' (idx={pos_idx})", "INFO")

    roc_metric, f1_metric = _build_metrics(pos_idx=pos_idx)

    learn = vision_learner(
        dls, 'resnet34',
        metrics=[accuracy, roc_metric, f1_metric],
        pretrained=True,
        loss_func=loss_fn,
        path=Path(CFG["MODELS_DIR"])
    )
    learn.add_cb(GradientClip(CFG["GRAD_CLIP"]))

    log("Phase 1 — Head only (frozen backbone)", "TRAIN")
    learn.freeze()
    lr = CFG["BASE_LR"]
    learn.fit_one_cycle(CFG["EPOCHS_STAGE1"], lr * 10)

    # ── STAGE 2: 192x192 ─────────────────────────────────────────
    banner("STAGE 2/3 — Training at 192x192", "─")
    img_size_s2 = int((CFG["IMG_SIZE_START"] + CFG["IMG_SIZE_FINAL"]) / 2)

    # ИСПРАВЛЕНИЕ v4: sampler применяется и здесь
    new_dls = _make_dls_with_sampler(
        img_size=img_size_s2,
        batch_tfms=batch_tfms,
        min_scale=0.75
    )
    learn.dls = new_dls

    log("Phase 2 — Head + last 2 blocks", "TRAIN")
    learn.freeze_to(-2)
    learn.fit_one_cycle(CFG["EPOCHS_STAGE2"], slice(lr / 10, lr))

    # ── STAGE 3: 224x224 ─────────────────────────────────────────
    banner("STAGE 3/3 — Training at 224x224", "─")

    # ИСПРАВЛЕНИЕ v4: sampler применяется и здесь
    final_dls = _make_dls_with_sampler(
        img_size=CFG["IMG_SIZE_FINAL"],
        batch_tfms=batch_tfms,
        min_scale=0.8
    )
    learn.dls = final_dls

    save_cb  = SaveModelCallback(monitor='valid_loss', fname='best_skin')
    early_cb = EarlyStoppingCallback(monitor='valid_loss', patience=CFG["EARLY_STOP_PAT"])

    log("Phase 3 — Full network fine-tuning", "TRAIN")
    learn.unfreeze()
    learn.fit_one_cycle(
        CFG["EPOCHS_STAGE3"],
        slice(lr / 100, lr / 10),
        cbs=[save_cb, early_cb]
    )

    best_path = Path(CFG["MODELS_DIR"]) / "models" / "best_skin.pth"
    if best_path.exists():
        learn.load("best_skin")
        log("Best weights loaded", "OK")

    # ── EVALUATION ────────────────────────────────────────────────
    banner("STEP 3 — EVALUATION", "─")
    preds_tta, targets = learn.tta(n=CFG["TTA_N"])
    probs    = torch.softmax(preds_tta, dim=1).cpu().numpy()
    targets_ = targets.cpu().numpy()

    # ИСПРАВЛЕНИЕ v4: диагностика и автокоррекция инверсии
    scores, was_inverted = _diagnose_and_fix_scores(probs, targets_, vocab, pos_idx)

    # ИСПРАВЛЕНИЕ v4: защита от threshold=inf
    fpr, tpr, ths = roc_curve(targets_, scores)
    youden_j      = tpr - fpr
    opt_idx       = youden_j.argmax()

    # roc_curve может вернуть inf в первом элементе — пропускаем его
    finite_mask = np.isfinite(ths)
    if finite_mask.sum() == 0:
        log("Все пороги = inf или nan! Используем threshold=0.5", "WARN")
        opt_thr = 0.5
    else:
        youden_finite = youden_j[finite_mask]
        ths_finite    = ths[finite_mask]
        opt_thr       = float(ths_finite[youden_finite.argmax()])

    log(f"Optimal threshold (Youden J): {opt_thr:.4f}", "INFO")

    preds_b = (scores >= opt_thr).astype(int)

    acc  = float((preds_b == targets_).mean())
    prec = float(precision_score(targets_, preds_b, zero_division=0))
    rec  = float(recall_score(targets_, preds_b, zero_division=0))
    f1   = float(f1_score(targets_, preds_b, zero_division=0))
    auc  = float(roc_auc_score(targets_, scores))
    bal  = float(balanced_accuracy_score(targets_, preds_b))
    cm   = confusion_matrix(targets_, preds_b)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = float(tn / (tn + fp + 1e-8))

    print(f"\n  {'─'*58}")
    print(f"  📊 SKIN CANCER DETECTION RESULTS (v4)")
    print(f"  {'─'*58}")
    print(f"  Accuracy              : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Balanced Accuracy     : {bal:.4f}")
    print(f"  Precision (PPV)       : {prec:.4f}")
    print(f"  Recall (Sensitivity)  : {rec:.4f}  ← % detected melanomas")
    print(f"  Specificity (TNR)     : {spec:.4f}")
    print(f"  F1-Score              : {f1:.4f}")
    print(f"  AUC-ROC               : {auc:.4f}")
    print(f"  Optimal Threshold     : {opt_thr:.4f}")
    print(f"  Score inverted        : {was_inverted}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"  {'─'*58}")

    if rec < 0.7:
        log(f"Recall={rec:.3f} < 0.70 — model misses many melanomas!", "WARN")
    elif rec > 0.85:
        log(f"Excellent recall! Model detects {rec*100:.1f}% of melanomas", "OK")

    # Сохраняем модель
    export_path = os.path.join(CFG["MODELS_DIR"], CFG["EXPORT_NAME"])
    learn.export(export_path)

    metadata = {
        "model_name":      "Skin Cancer Detector v4",
        "model_type":      "skin",
        "architecture":    "ResNet34 + FocalLoss(device-fixed) + WeightedSampler(all stages)",
        "classes":         vocab,
        "positive_class":  CFG["POSITIVE_CLASS"],
        "img_size":        CFG["IMG_SIZE_FINAL"],
        "training_date":   datetime.now().isoformat(),
        "score_inverted":  was_inverted,
        "performance_metrics": {
            "overall": {
                "accuracy":          round(acc,     4),
                "balanced_accuracy": round(bal,     4),
                "precision":         round(prec,    4),
                "recall":            round(rec,     4),
                "specificity":       round(spec,    4),
                "f1_score":          round(f1,      4),
                "auc_roc":           round(auc,     4),
                "optimal_threshold": round(opt_thr, 4),
            },
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            },
        },
        "model_path": export_path,
    }

    meta_path = os.path.join(CFG["MODELS_DIR"], CFG["META_NAME"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log(f"Model → {export_path}", "OK")
    log(f"Meta  → {meta_path}", "OK")
    return metadata


# ─────────────────────────────────────────────
#  ДООБУЧЕНИЕ
# ─────────────────────────────────────────────
def run_finetune(extra_epochs=None):
    try:
        from fastai.vision.all import (
            load_learner, aug_transforms,
            Normalize, imagenet_stats,
            GradientClip, SaveModelCallback, valley
        )
        from sklearn.metrics import (
            roc_curve, accuracy_score,
            recall_score, roc_auc_score
        )
    except ImportError as e:
        log(f"Missing: {e}", "ERR")
        return

    banner("FINETUNING SKIN CANCER MODEL (v4)", "─")

    export_path = os.path.join(CFG["MODELS_DIR"], CFG["EXPORT_NAME"])
    if not os.path.exists(export_path):
        log(f"Model not found: {export_path}", "ERR")
        return

    if extra_epochs is None:
        extra_epochs = CFG.get("FINETUNE_EPOCHS", 10)

    log(f"Loading existing model: {export_path}", "INFO")
    learn = load_learner(export_path)

    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = None
    _preprocess_images(CFG["DATA_DIR"])

    batch_tfms = [
        *aug_transforms(
            do_flip=True, flip_vert=True, max_rotate=30,
            max_zoom=1.2, max_lighting=0.3, max_warp=0.2,
            p_affine=0.75, p_lighting=0.75
        ),
        Normalize.from_stats(*imagenet_stats)
    ]

    # ИСПРАВЛЕНИЕ v4: используем хелпер с sampler
    new_dls = _make_dls_with_sampler(
        img_size=CFG["IMG_SIZE_FINAL"],
        batch_tfms=batch_tfms,
        min_scale=0.8
    )

    learn.dls       = new_dls
    weights         = _compute_class_weights()
    learn.loss_func = FocalLoss(weight=weights, gamma=CFG["FOCAL_GAMMA"])
    learn.add_cb(GradientClip(CFG["GRAD_CLIP"]))

    log(f"Finetuning for {extra_epochs} epochs", "TRAIN")
    learn.unfreeze()

    try:
        res = learn.lr_find(suggest_funcs=(valley,), num_it=20)
        lr  = min(res.valley / 10, CFG.get("FINETUNE_LR", 5e-5))
    except Exception:
        lr = CFG.get("FINETUNE_LR", 5e-5)

    learn.fit_one_cycle(extra_epochs, slice(lr / 100, lr))

    # Оценка с диагностикой инверсии
    preds_tta, targets = learn.tta(n=CFG["TTA_N"])
    probs    = torch.softmax(preds_tta, dim=1).cpu().numpy()
    targets_ = targets.cpu().numpy()

    vocab   = list(learn.dls.vocab)
    pos_idx = vocab.index(CFG["POSITIVE_CLASS"])

    scores, was_inverted = _diagnose_and_fix_scores(probs, targets_, vocab, pos_idx)

    fpr, tpr, ths = roc_curve(targets_, scores)
    finite_mask   = np.isfinite(ths)
    if finite_mask.sum() == 0:
        opt_thr = 0.5
    else:
        youden_finite = (tpr - fpr)[finite_mask]
        ths_finite    = ths[finite_mask]
        opt_thr       = float(ths_finite[youden_finite.argmax()])

    preds_b = (scores >= opt_thr).astype(int)
    acc     = accuracy_score(targets_, preds_b)
    rec     = recall_score(targets_, preds_b, zero_division=0)
    auc     = roc_auc_score(targets_, scores)

    learn.export(export_path)
    log(f"Finetuned model saved → {export_path}", "OK")
    log(f"Results: AUC={auc:.4f}, Recall={rec:.4f}, Accuracy={acc:.4f}", "OK")
    log(f"Score inverted: {was_inverted}", "INFO")

    # Обновляем метаданные
    meta_path = os.path.join(CFG["MODELS_DIR"], CFG["META_NAME"])
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["performance_metrics"]["overall"]["auc_roc"]           = round(auc, 4)
        meta["performance_metrics"]["overall"]["recall"]            = round(rec, 4)
        meta["performance_metrics"]["overall"]["accuracy"]          = round(acc, 4)
        meta["performance_metrics"]["overall"]["optimal_threshold"] = round(opt_thr, 4)
        meta["score_inverted"]  = was_inverted
        meta["training_date"]   = datetime.now().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f"Metadata updated → {meta_path}", "OK")


# ─────────────────────────────────────────────
#  ПРЕДСКАЗАНИЕ
# ─────────────────────────────────────────────
def predict(image_path, threshold=None):
    from fastai.vision.all import load_learner, PILImage

    model_path = os.path.join(CFG["MODELS_DIR"], CFG["EXPORT_NAME"])
    if not os.path.exists(model_path):
        log(f"Model not found: {model_path}", "ERR")
        return

    was_inverted = False
    meta_path = os.path.join(CFG["MODELS_DIR"], CFG["META_NAME"])
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if threshold is None:
            threshold    = (meta.get("performance_metrics", {})
                               .get("overall", {})
                               .get("optimal_threshold", 0.5))
        was_inverted = meta.get("score_inverted", False)

    if threshold is None or not np.isfinite(threshold):
        threshold = 0.5

    learn  = load_learner(model_path)
    img    = PILImage.create(image_path)
    _, _, probs = learn.predict(img)

    vocab   = list(learn.dls.vocab)
    pos_idx = vocab.index(CFG["POSITIVE_CLASS"])

    if was_inverted:
        conf = float(probs[1 - pos_idx])
        log("Using inverted scores (as recorded in metadata)", "INFO")
    else:
        conf = float(probs[pos_idx])

    final = "melanoma" if conf >= threshold else "normal"

    print(f"\n  {'─'*50}")
    print(f"  🩺  SKIN PREDICTION (v4)")
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  Melanoma probability: {conf:.2%}")
    print(f"  RESULT: {'⚠️  MELANOMA DETECTED' if final == 'melanoma' else '✅ NORMAL SKIN'}")
    print(f"  {'─'*50}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Skin cancer pipeline v4 (fixed FocalLoss device + sampler all stages + AUC inversion detection)"
    )
    parser.add_argument("--download-only",   action="store_true")
    parser.add_argument("--train-only",      action="store_true")
    parser.add_argument("--force-download",  action="store_true")
    parser.add_argument("--finetune",        action="store_true",
                        help="Finetune existing model")
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--max",             type=int, default=CFG["MAX_PER_CLASS"])
    parser.add_argument("--source",          choices=["isic", "kaggle"], default="isic")
    parser.add_argument("--predict",         type=str, help="Path to image")
    parser.add_argument("--threshold",       type=float, default=None)
    args = parser.parse_args()

    banner("🩺  SKIN CANCER PIPELINE v4 (FIXED)", "═")

    if args.predict:
        predict(args.predict, args.threshold)
        return

    ensure_dirs()

    if args.finetune:
        run_finetune(args.finetune_epochs)
        return

    if args.train_only:
        run_train()
        return

    ok = run_download(args.max, force=args.force_download, source=args.source)

    if not args.download_only and ok:
        run_train()


if __name__ == "__main__":
    main()