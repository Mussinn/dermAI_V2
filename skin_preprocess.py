"""
skin_preprocess.py — Предобработка дерматоскопических снимков
═══════════════════════════════════════════════════════════════
Специфика кожных снимков vs рентгенов:
  ✓ ЦВЕТ важен (меланома имеет характерные цветовые паттерны)
  ✓ Волосы — главный артефакт (DullRazor алгоритм)
  ✓ Тёмный виньетинг по краям (дерматоскоп)
  ✓ Пузырьки воздуха (гелевые артефакты)
  ✓ Цветовые маркеры (линейки, точки)
  ✗ НЕТ чёрных полей как в рентгене
  ✗ НЕ применять CLAHE к цветному изображению напрямую (только к L-каналу)

Порядок шагов:
  1. Удаление виньетинга (тёмные края дерматоскопа)
  2. Удаление волос (DullRazor)
  3. Удаление пузырьков и маркеров
  4. Нормализация цвета между датасетами (Reinhard или percentile)
  5. Усиление контраста (CLAHE только по L-каналу LAB)
  6. Resize с сохранением пропорций

Интеграция:
    from skin_preprocess import preprocess_dataset

    # Единоразово перед обучением
    preprocess_dataset("skin_data", "skin_data_processed")

    # В skin_pipeline.py:
    CFG["DATA_DIR"] = "skin_data_processed"

Требования:
    pip install opencv-python-headless numpy Pillow tqdm
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  ШАГ 1: УДАЛЕНИЕ ВИНЬЕТИНГА (тёмный круг дерматоскопа)
#  Проблема: дерматоскоп даёт тёмный круглый/овальный ободок
#  по краям. Модель учится на форме ободка вместо паттернов кожи.
#  Решение: детектируем круговую тёмную маску и crop по ней.
# ══════════════════════════════════════════════════════════════════

def remove_vignette(img_bgr: np.ndarray) -> np.ndarray:
    """
    Убирает тёмный ободок дерматоскопа.

    Алгоритм:
      1. Конвертируем в grayscale
      2. Пороговая обработка — находим тёмные края
      3. Ищем крупнейшую светлую область (сама кожа)
      4. Crop по bounding box этой области

    Parameters
    ----------
    img_bgr : цветное BGR изображение

    Returns
    -------
    Обрезанное BGR изображение.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Размытие — убираем шум и волосы
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)

    # Отсекаем тёмные пиксели (виньетинг < 15% от максимума)
    threshold = blurred.max() * 0.15
    mask = (blurred > threshold).astype(np.uint8) * 255

    # Морфологическое закрытие — заполняем разрывы
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Bounding box ненулевой области
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return img_bgr

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Проверка адекватности
    crop_h = r_max - r_min
    crop_w = c_max - c_min
    if crop_h < h * 0.4 or crop_w < w * 0.4:
        return img_bgr  # виньетинг занял слишком много — не трогаем

    return img_bgr[r_min:r_max+1, c_min:c_max+1]


# ══════════════════════════════════════════════════════════════════
#  ШАГ 2: УДАЛЕНИЕ ВОЛОС (DullRazor алгоритм)
#  Проблема: волосы перекрывают поражения кожи и создают
#  артефактные тёмные линии которые модель принимает за паттерны.
#  DullRazor — стандарт для дерматоскопии (Lee et al. 1997).
# ══════════════════════════════════════════════════════════════════

def remove_hair(img_bgr: np.ndarray,
                kernel_size: int = 17,
                threshold: int = 10) -> np.ndarray:
    """
    DullRazor алгоритм удаления волос.

    Алгоритм:
      1. Grayscale → морфологическое закрытие (blackhat)
         → выделяем тёмные тонкие структуры (волосы)
      2. Пороговая обработка → маска волос
      3. Inpainting — восстанавливаем кожу под волосами

    Parameters
    ----------
    kernel_size : размер ядра (17–25 для дерматоскопии)
                  больше = убирает более толстые волосы
    threshold   : чувствительность (5–15, меньше = агрессивнее)

    Returns
    -------
    BGR изображение без волос.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blackhat: выделяет тёмные объекты на светлом фоне
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Маска волос
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # Морфологическое расширение — чуть расширяем маску
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    # Нет волос — возвращаем оригинал
    if hair_mask.sum() == 0:
        return img_bgr

    # Inpainting — восстанавливаем кожу
    result = cv2.inpaint(img_bgr, hair_mask,
                         inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    return result


# ══════════════════════════════════════════════════════════════════
#  ШАГ 3: УДАЛЕНИЕ ПУЗЫРЬКОВ И МАРКЕРОВ
#  Проблема: гелевые пузырьки (белые круглые области),
#  цветные маркеры (точки, линейки), артефакты давления.
#  Решение: детектируем аномально яркие/тёмные округлые области.
# ══════════════════════════════════════════════════════════════════

def remove_bubbles_and_markers(img_bgr: np.ndarray,
                                bright_pct: float = 98.5,
                                dark_pct: float = 1.5,
                                min_area: int = 30,
                                max_area: int = 3000) -> np.ndarray:
    """
    Удаляет пузырьки воздуха (белые), тёмные маркеры и цветные точки.

    Parameters
    ----------
    bright_pct : перцентиль для определения "слишком белого" (пузырьки)
    dark_pct   : перцентиль для определения "слишком тёмного" (маркеры)
    min_area   : минимальная площадь артефакта (px²)
    max_area   : максимальная площадь (px²) — крупнее это уже кожа
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    result = img_bgr.copy()

    for pct, comparison in [(bright_pct, "bright"), (dark_pct, "dark")]:
        threshold_val = np.percentile(gray, pct)

        if comparison == "bright":
            mask = (gray >= threshold_val).astype(np.uint8) * 255
        else:
            mask = (gray <= threshold_val).astype(np.uint8) * 255

        # Морфология — убираем шум
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Связные компоненты
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        artifact_mask = np.zeros_like(gray, dtype=np.uint8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                artifact_mask[labels == label_id] = 255

        if artifact_mask.sum() == 0:
            continue

        # Расширяем маску
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        artifact_mask = cv2.dilate(artifact_mask, kernel_d, iterations=1)

        result = cv2.inpaint(result, artifact_mask,
                             inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return result


# ══════════════════════════════════════════════════════════════════
#  ШАГ 4: НОРМАЛИЗАЦИЯ ЦВЕТА МЕЖДУ ДАТАСЕТАМИ
#  Проблема: ISIC, HAM10000, Kaggle — разные камеры, разный баланс
#  белого, разная яркость. Модель будет учиться на цвете камеры
#  вместо цвета поражения.
#
#  Решение: Reinhard color normalization — приводим к эталонным
#  статистикам цвета (среднее и std каждого канала LAB).
#  Это стандарт в computational pathology.
# ══════════════════════════════════════════════════════════════════

# Эталонные статистики LAB для дерматоскопии
# Вычислены на репрезентативной выборке HAM10000
SKIN_TARGET_STATS = {
    "L": {"mean": 168.0, "std": 35.0},
    "A": {"mean": 128.5, "std": 8.0},
    "B": {"mean": 135.0, "std": 12.0},
}


def normalize_color_reinhard(img_bgr: np.ndarray,
                              target_stats: dict = None) -> np.ndarray:
    """
    Reinhard нормализация цвета в пространстве LAB.

    Приводит статистики (mean, std) каждого канала к эталонным.
    Сохраняет структуру изображения, меняет только цветовой тон.

    Parameters
    ----------
    img_bgr      : BGR изображение
    target_stats : словарь с {"L": {"mean":..., "std":...}, "A":..., "B":...}
                   None = используем SKIN_TARGET_STATS

    Returns
    -------
    Цветонормализованное BGR изображение.
    """
    if target_stats is None:
        target_stats = SKIN_TARGET_STATS

    # BGR → LAB (float32)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    channel_names = ["L", "A", "B"]
    for i, ch_name in enumerate(channel_names):
        channel = img_lab[:, :, i]
        src_mean = channel.mean()
        src_std  = channel.std() + 1e-6

        tgt_mean = target_stats[ch_name]["mean"]
        tgt_std  = target_stats[ch_name]["std"]

        # Нормализация: (x - src_mean) / src_std * tgt_std + tgt_mean
        normalized = (channel - src_mean) / src_std * tgt_std + tgt_mean
        img_lab[:, :, i] = normalized

    # Clip и конвертация обратно
    img_lab = np.clip(img_lab, 0, 255).astype(np.uint8)
    result  = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return result


# ══════════════════════════════════════════════════════════════════
#  ШАГ 5: УСИЛЕНИЕ КОНТРАСТА (CLAHE только по L-каналу)
#  ВАЖНО: для кожи НЕ применяем CLAHE к RGB напрямую —
#  это сдвигает цветовой баланс и уничтожает цветовые признаки
#  меланомы. Только по L-каналу (яркость) в LAB пространстве.
#
#  Параметры:
#    clipLimit=1.5 — мягче чем для рентгена (кожа менее контрастная)
#    tileGridSize=(8,8) — стандарт для дерматоскопии
# ══════════════════════════════════════════════════════════════════

def apply_clahe_skin(img_bgr: np.ndarray,
                     clip_limit: float = 1.5,
                     tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE только по L-каналу LAB — сохраняет цвета, улучшает контраст.

    НЕ используй cv2.createCLAHE() напрямую к RGB — это меняет цвет!

    Parameters
    ----------
    clip_limit : 1.0–2.0 для кожи (меньше чем для рентгена!)
    tile_grid  : (8,8) для изображений 224–512px
    """
    # BGR → LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # CLAHE только к L (яркость)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    # Собираем обратно
    img_lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


# ══════════════════════════════════════════════════════════════════
#  ШАГ 6: ЦЕНТРАЛЬНЫЙ CROP (дерматоскопия — поражение в центре)
#  Дополнительный шаг: дерматоскопические снимки всегда центрируют
#  поражение. Небольшой central crop убирает края с артефактами.
# ══════════════════════════════════════════════════════════════════

def center_crop_skin(img_bgr: np.ndarray,
                     crop_pct: float = 0.90) -> np.ndarray:
    """
    Центральный crop: берём центральные crop_pct% изображения.

    Убирает крайние области с виньетингом и артефактами давления.

    Parameters
    ----------
    crop_pct : доля сохраняемой области (0.85–0.95)
               0.90 = убираем 5% с каждого края
    """
    h, w = img_bgr.shape[:2]
    margin_h = int(h * (1 - crop_pct) / 2)
    margin_w = int(w * (1 - crop_pct) / 2)

    if margin_h == 0 and margin_w == 0:
        return img_bgr

    return img_bgr[margin_h:h-margin_h, margin_w:w-margin_w]


# ══════════════════════════════════════════════════════════════════
#  ГЛАВНЫЙ КЛАСС: SkinPreprocessor
# ══════════════════════════════════════════════════════════════════

class SkinPreprocessor:
    """
    Полный pipeline предобработки дерматоскопических снимков.

    Порядок шагов (оптимален для дерматоскопии):
      1. Загрузка как RGB (цвет критически важен!)
      2. Удаление виньетинга (тёмный ободок дерматоскопа)
      3. Центральный crop (убираем края)
      4. Удаление волос (DullRazor)
      5. Удаление пузырьков и маркеров
      6. Нормализация цвета (Reinhard LAB)
      7. CLAHE по L-каналу
      8. Resize до target_size

    Пример использования:
        prep = SkinPreprocessor(target_size=256)
        pil_img = prep("path/to/lesion.jpg")

        # Батчем:
        prep.process_directory("skin_data/melanoma", "skin_data_proc/melanoma")
    """

    def __init__(
        self,
        target_size: int = 256,
        # Что включать
        do_vignette: bool = True,
        do_center_crop: bool = True,
        do_hair: bool = True,
        do_bubbles: bool = True,
        do_color_norm: bool = True,
        do_clahe: bool = True,
        # Параметры волос
        hair_kernel: int = 17,
        hair_threshold: int = 10,
        # Параметры пузырьков
        bubble_bright_pct: float = 98.5,
        bubble_dark_pct: float = 1.5,
        bubble_min_area: int = 30,
        bubble_max_area: int = 3000,
        # Параметры CLAHE
        clahe_clip: float = 1.5,
        clahe_grid: tuple = (8, 8),
        # Central crop
        crop_pct: float = 0.90,
        # Нормализация цвета
        color_target_stats: dict = None,
    ):
        self.target_size = target_size
        self.do_vignette = do_vignette
        self.do_center_crop = do_center_crop
        self.do_hair = do_hair
        self.do_bubbles = do_bubbles
        self.do_color_norm = do_color_norm
        self.do_clahe = do_clahe
        self.hair_kernel = hair_kernel
        self.hair_threshold = hair_threshold
        self.bubble_bright_pct = bubble_bright_pct
        self.bubble_dark_pct = bubble_dark_pct
        self.bubble_min_area = bubble_min_area
        self.bubble_max_area = bubble_max_area
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.crop_pct = crop_pct
        self.color_target_stats = color_target_stats or SKIN_TARGET_STATS

    def load_as_bgr(self, path) -> np.ndarray:
        """Загружает изображение как BGR uint8 (для OpenCV)."""
        path = str(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            # Запасной вариант через PIL
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # Конвертируем если нужно
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)

        return img

    def process(self, path) -> Image.Image:
        """
        Обрабатывает один снимок и возвращает PIL Image (RGB).

        Parameters
        ----------
        path : str или Path

        Returns
        -------
        PIL.Image в режиме "RGB", размером target_size x target_size
        """
        img = self.load_as_bgr(path)

        # ── 1. Удаление виньетинга ───────────
        if self.do_vignette:
            img = remove_vignette(img)

        # ── 2. Центральный crop ──────────────
        if self.do_center_crop:
            img = center_crop_skin(img, self.crop_pct)

        # ── 3. Удаление волос ─────────────────
        if self.do_hair:
            img = remove_hair(img, self.hair_kernel, self.hair_threshold)

        # ── 4. Удаление пузырьков/маркеров ───
        if self.do_bubbles:
            img = remove_bubbles_and_markers(
                img,
                self.bubble_bright_pct,
                self.bubble_dark_pct,
                self.bubble_min_area,
                self.bubble_max_area,
            )

        # ── 5. Нормализация цвета ─────────────
        if self.do_color_norm:
            img = normalize_color_reinhard(img, self.color_target_stats)

        # ── 6. CLAHE (только L-канал) ─────────
        if self.do_clahe:
            img = apply_clahe_skin(img, self.clahe_clip, self.clahe_grid)

        # ── 7. Resize ────────────────────────
        img = cv2.resize(img, (self.target_size, self.target_size),
                         interpolation=cv2.INTER_LANCZOS4)

        # ── 8. BGR → RGB для PIL/fastai ──────
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def __call__(self, path) -> Image.Image:
        return self.process(path)

    def process_directory(
        self,
        src_dir: str,
        dst_dir: str,
        n_workers: int = 4,
        overwrite: bool = False,
    ) -> dict:
        """
        Обрабатывает все снимки в папке.

        Parameters
        ----------
        src_dir   : папка с исходными снимками
        dst_dir   : папка для сохранения
        n_workers : потоков (4–8 оптимально)
        overwrite : перезаписывать существующие
        """
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [f for f in src_dir.iterdir() if f.suffix.lower() in exts]

        if not files:
            print(f"  ⚠️  No images found in {src_dir}")
            return {"ok": 0, "fail": 0, "skipped": 0}

        stats = {"ok": 0, "fail": 0, "skipped": 0}
        t0 = time.time()

        def _process_one(src_path):
            dst_path = dst_dir / (src_path.stem + ".jpg")
            if not overwrite and dst_path.exists():
                return "skipped"
            try:
                pil_img = self.process(src_path)
                pil_img.save(dst_path, "JPEG", quality=95)
                return "ok"
            except Exception as e:
                return f"fail:{e}"

        print(f"  ⬇️  Processing {len(files)} images from {src_dir.name}/")
        print(f"      → {dst_dir}")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one, f): f for f in files}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result == "ok":
                    stats["ok"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["fail"] += 1
                    # Показываем первые 3 ошибки
                    if stats["fail"] <= 3:
                        src_path = futures[future]
                        print(f"      ❌ {src_path.name}: {result}")
                if done % 100 == 0 or done == len(files):
                    elapsed = time.time() - t0
                    speed = done / max(elapsed, 0.1)
                    print(f"      {done}/{len(files)}  "
                          f"ok={stats['ok']} skip={stats['skipped']} fail={stats['fail']}  "
                          f"({speed:.1f} img/s)", flush=True)

        elapsed = time.time() - t0
        print(f"  ✅  Done in {elapsed:.1f}s  "
              f"({stats['ok']} ok, {stats['fail']} fail, {stats['skipped']} skipped)")
        return stats


# ══════════════════════════════════════════════════════════════════
#  ФУНКЦИЯ-ОБЁРТКА: preprocess_dataset
# ══════════════════════════════════════════════════════════════════

def preprocess_dataset(
    src_root: str = "skin_data",
    dst_root: str = "skin_data_processed",
    target_size: int = 256,
    n_workers: int = 4,
    do_hair: bool = True,
    do_vignette: bool = True,
    do_color_norm: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Единоразовая предобработка всего датасета кожи.

    Вызывай один раз перед обучением:
        preprocess_dataset("skin_data", "skin_data_processed")

    Затем в skin_pipeline.py:
        CFG["DATA_DIR"] = "skin_data_processed"

    Parameters
    ----------
    src_root      : папка с melanoma/ и normal/
    dst_root      : куда сохранять обработанные снимки
    target_size   : размер (рекомендуется IMG_SIZE + 32 = 256)
    n_workers     : потоков (4 на CPU, 8 если SSD)
    do_hair       : удалять волосы (рекомендуется True)
    do_vignette   : убирать виньетинг (рекомендуется True)
    do_color_norm : нормализация цвета (рекомендуется True)
    overwrite     : перезаписывать файлы
    """
    print("\n" + "═" * 62)
    print("  🩺  SKIN PREPROCESSING PIPELINE")
    print("═" * 62)
    print(f"  Волосы      : {'ON' if do_hair else 'OFF'}")
    print(f"  Виньетинг   : {'ON' if do_vignette else 'OFF'}")
    print(f"  Цвет (Rein) : {'ON' if do_color_norm else 'OFF'}")
    print(f"  CLAHE (L)   : ON (всегда)")
    print(f"  Пузырьки    : ON (всегда)")
    print(f"  Target size : {target_size}px")
    print()

    preprocessor = SkinPreprocessor(
        target_size=target_size,
        do_vignette=do_vignette,
        do_center_crop=True,
        do_hair=do_hair,
        do_bubbles=True,
        do_color_norm=do_color_norm,
        do_clahe=True,
        clahe_clip=1.5,
        clahe_grid=(8, 8),
        crop_pct=0.90,
    )

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    class_dirs = [d for d in src_root.iterdir()
                  if d.is_dir() and not d.name.startswith(".")]

    if not class_dirs:
        print(f"  ❌  No class folders found in {src_root}")
        return

    total_stats = {"ok": 0, "fail": 0, "skipped": 0}

    for class_dir in sorted(class_dirs):
        dst_class_dir = dst_root / class_dir.name
        stats = preprocessor.process_directory(
            src_dir=class_dir,
            dst_dir=dst_class_dir,
            n_workers=n_workers,
            overwrite=overwrite,
        )
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"\n  📊  TOTAL: ok={total_stats['ok']}  "
          f"fail={total_stats['fail']}  skipped={total_stats['skipped']}")
    print(f"  ✅  Saved to: {dst_root}/")
    print("\n  В skin_pipeline.py измени одну строку:")
    print(f'    CFG["DATA_DIR"] = "{dst_root}"')
    print("═" * 62 + "\n")


# ══════════════════════════════════════════════════════════════════
#  ДИАГНОСТИКА: визуализация шагов
# ══════════════════════════════════════════════════════════════════

def visualize_steps(image_path: str, save_path: str = "skin_preprocess_debug.jpg"):
    """
    Сохраняет коллаж — каждый шаг предобработки по отдельности.

    Запуск:
        python skin_preprocess.py --viz path/to/lesion.jpg

    Смотри на:
      - Шаг 2 (волосы): убраны ли волосы без артефактов?
      - Шаг 4 (цвет): не слишком ли изменился цвет?
      - Шаг 5 (CLAHE): улучшился ли контраст границ поражения?
    """
    prep = SkinPreprocessor(target_size=512)
    img = prep.load_as_bgr(image_path)

    steps_bgr = []

    # Оригинал
    steps_bgr.append(("Original", img.copy()))

    # После виньетинга
    img_v = remove_vignette(img)
    steps_bgr.append(("1. Remove vignette", img_v.copy()))

    # После центрального crop
    img_c = center_crop_skin(img_v)
    steps_bgr.append(("2. Center crop", img_c.copy()))

    # После волос
    img_h = remove_hair(img_c)
    steps_bgr.append(("3. Remove hair", img_h.copy()))

    # После пузырьков
    img_b = remove_bubbles_and_markers(img_h)
    steps_bgr.append(("4. Remove bubbles", img_b.copy()))

    # После нормализации цвета
    img_n = normalize_color_reinhard(img_b)
    steps_bgr.append(("5. Color norm", img_n.copy()))

    # После CLAHE
    img_cl = apply_clahe_skin(img_n)
    steps_bgr.append(("6. CLAHE (L only)", img_cl.copy()))

    # Рисуем коллаж
    target_h, target_w = 400, 400
    n = len(steps_bgr)
    cols = 4
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * (target_h + 30), cols * target_w, 3), dtype=np.uint8)
    canvas[:] = 30  # тёмный фон

    for i, (title, step_img) in enumerate(steps_bgr):
        row = i // cols
        col = i % cols
        y = row * (target_h + 30)
        x = col * target_w

        resized = cv2.resize(step_img, (target_w, target_h),
                             interpolation=cv2.INTER_LANCZOS4)
        canvas[y:y+target_h, x:x+target_w] = resized

        cv2.putText(canvas, title,
                    (x + 5, y + target_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imwrite(save_path, canvas)
    print(f"  ✅  Debug visualization saved → {save_path}")
    print(f"  Проверь: убраны ли волосы? сохранён ли цвет меланомы?")


# ══════════════════════════════════════════════════════════════════
#  MAIN — CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Skin Lesion Preprocessing Pipeline"
    )
    parser.add_argument("--src",          default="skin_data",
                        help="Исходная папка с данными")
    parser.add_argument("--dst",          default="skin_data_processed",
                        help="Папка для обработанных снимков")
    parser.add_argument("--size",         type=int, default=256,
                        help="Размер выходного изображения (default=256)")
    parser.add_argument("--workers",      type=int, default=4,
                        help="Число потоков (default=4)")
    parser.add_argument("--no-hair",      action="store_true",
                        help="Отключить удаление волос")
    parser.add_argument("--no-vignette",  action="store_true",
                        help="Отключить удаление виньетинга")
    parser.add_argument("--no-color",     action="store_true",
                        help="Отключить нормализацию цвета")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Перезаписывать существующие файлы")
    parser.add_argument("--viz",          type=str, default=None,
                        help="Путь к снимку для визуализации шагов")
    args = parser.parse_args()

    if args.viz:
        visualize_steps(args.viz)
    else:
        preprocess_dataset(
            src_root=args.src,
            dst_root=args.dst,
            target_size=args.size,
            n_workers=args.workers,
            do_hair=not args.no_hair,
            do_vignette=not args.no_vignette,
            do_color_norm=not args.no_color,
            overwrite=args.overwrite,
        )