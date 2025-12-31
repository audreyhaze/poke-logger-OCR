import cv2
import string
import json
import glob
import os
from PIL import Image
import pytesseract
import re
import csv
import time
import numpy as np
from threading import Thread
from queue import Queue
from skimage.metrics import structural_similarity as ssim

# Path to your tesseract exe
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------
# Utilities: config loaders
# -------------------------
def list_game_configs(folder="configs"):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")]

def choose_config(config_files):
    if not config_files:
        raise FileNotFoundError("No config files found.")
    if len(config_files) == 1:
        print(f"Only one config found, auto-selecting: {config_files[0]}")
        return config_files[0]

    print("Available configurations:")
    for i, file in enumerate(config_files):
        print(f"{i+1}. {file}")
    while True:
        choice = input("Select a config number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(config_files):
            return config_files[int(choice)-1]
        print("Invalid choice.")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_layout(path):
    """Load layout JSON mapping (name -> [left, top, w, h])."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Layout file not found: {path}")
    config = load_json(path)
    # Convert lists to toBox tuples
    return {key: toBox(*vals) for key, vals in config.items()}

def load_images_from_folder(folder):
    """Return dict {basename: image} for all png and jpg in folder."""
    images = {}
    if not os.path.isdir(folder):
        return images
    for file in glob.glob(os.path.join(folder, "*")):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            key = os.path.splitext(os.path.basename(file))[0]
            img = cv2.imread(file)
            if img is None:
                print(f"[ERROR] Failed to load image file: {file}")
            images[key] = img
    return images

def load_templates_from_json_map(json_path):
    """Load mapping JSON where keys -> path to image files."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Template config not found: {json_path}")
    mapping = load_json(json_path)
    templates = {}
    for name, path in mapping.items():
        if not os.path.exists(path):
            print(f"[WARNING] Template not found: {path} (key: {name})")
            templates[name] = None
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Failed to read template image: {path} (key: {name})")
        templates[name] = img
    return templates

def load_templates_auto(path_or_folder):
    """If JSON file -> load mapping; if folder -> load all images; else error."""
    if os.path.isdir(path_or_folder):
        return load_images_from_folder(path_or_folder)
    if os.path.isfile(path_or_folder) and path_or_folder.lower().endswith(".json"):
        return load_templates_from_json_map(path_or_folder)
    raise FileNotFoundError(f"No template folder or JSON mapping found at: {path_or_folder}")

def load_ball_templates(config_path):
    try:
        return load_templates_auto(config_path)
    except Exception as e:
        print(f"[ERROR] load_ball_templates: {e}")
        return {}

def load_origin_templates(config_path):
    try:
        return load_templates_auto(config_path)
    except Exception as e:
        print(f"[ERROR] load_origin_templates: {e}")
        return {}

def load_main_config(path):
    return load_json(path)

def parse_color_string(s):
    """Convert '255,77,62' or [255,77,62] -> (255,77,62)."""
    if s is None:
        raise ValueError("Color value is None")
    if isinstance(s, str):
        parts = s.split(",")
        return tuple(int(p.strip()) for p in parts)
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    raise ValueError(f"Invalid color format: {s}")

def parse_xy_string(s):
    """Convert '12,22' or [12,22] or (12,22) -> (12,22)."""
    if s is None:
        raise ValueError("XY value is None")
    if isinstance(s, str):
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid XY string (needs 2 values): {s}")
        return tuple(int(p.strip()) for p in parts)
    if isinstance(s, (list, tuple)):
        if len(s) != 2:
            raise ValueError(f"Invalid XY list/tuple (needs 2 values): {s}")
        return tuple(int(v) for v in s)
    raise ValueError(f"Invalid XY format: {s}")

def load_full_configuration(main_config):
    """
    Loads layout/ball/origin plus gender settings from main config.
    Returns:
       regions, ball_templates, origin_templates, male_color, female_color, gender_tolerance, gender_sample_pos
    """
    layout_path = main_config.get("layout")
    ball_path   = main_config.get("ball")
    origin_path = main_config.get("origin")
    
    # sensible defaults if not provided
    male_color_raw   = main_config.get("male_color", "100,128,255")
    female_color_raw = main_config.get("female_color", "255,77,62")
    gender_tolerance = main_config.get("gender_tolerance", 30)
    gender_sample_pos_raw = main_config.get("gender_sample_pos", "12,22")

    # parse
    try:
        male_color   = parse_color_string(male_color_raw)
    except Exception as e:
        print(f"[WARN] invalid male_color in config: {male_color_raw} -> using default (100,128,255). Error: {e}")
        male_color = (100,128,255)

    try:
        female_color = parse_color_string(female_color_raw)
    except Exception as e:
        print(f"[WARN] invalid female_color in config: {female_color_raw} -> using default (255,77,62). Error: {e}")
        female_color = (255,77,62)

    try:
        gender_sample_pos = parse_xy_string(gender_sample_pos_raw)
    except Exception as e:
        print(f"[WARN] invalid gender_sample_pos in config: {gender_sample_pos_raw} -> using default (12,22). Error: {e}")
        gender_sample_pos = (12,22)

    # validate tolerance
    try:
        gender_tolerance = int(gender_tolerance)
    except Exception:
        gender_tolerance = 30

    if layout_path is None or ball_path is None or origin_path is None:
        raise KeyError("Main config must include 'layout', 'ball', and 'origin' keys.")

    regions = load_layout(layout_path)
    ball_templates = load_ball_templates(ball_path)
    origin_templates = load_origin_templates(origin_path)

    return regions, ball_templates, origin_templates, male_color, female_color, gender_tolerance, gender_sample_pos

# -------------------------
# Helper / image utilities
# -------------------------
def toBox(left, top, width, height):
    """Convert (left, top, width, height) → (left, top, right, bottom)"""
    return (left, top, left + width, top + height)

def preprocess_for_ocr(crop_img):
    gray = crop_img.convert('L')  # grayscale
    bw = gray.point(lambda x: 0 if x < 128 else 255, '1')  # simple threshold
    return bw

def cleanAllButDate(text):
    if not text:
        return "NODATEFOUND"

    # ---- Strip leading zeros ONLY on the first date segment ----
    # Example: "011/26/2022" → "11/26/2022"
    text = re.sub(
        r"\b0+(\d{1,2}/\d{1,2}/\d{4})",
        lambda m: m.group(1).lstrip("0"),
        text
    )
    
    pattern = r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b"
    match = re.search(pattern, text)
    return match.group() if match else "NODATEFOUND"

def color_close(c1, c2, tol):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(c1, c2))

def checkGenderColor(img, male_color=(100,128,255), female_color=(255,77,62), tolerance=30, samplePos=(12,22)):
    """img is a PIL image crop. Return 'M', 'F' or ''. """
    try:
        pixel = img.getpixel(samplePos)
    except Exception:
        return ""
    if color_close(pixel, male_color, tolerance):
        return "M"
    if color_close(pixel, female_color, tolerance):
        return "F"
    return ""

# -------------------------
# Template matching helpers
# -------------------------
def compare_ball(img, template):
    """Compare icon vs template using SSIM + HSV histogram"""
    if img is None or template is None:
        return 0.0  # low score
    try:
        img = cv2.resize(img, (template.shape[1], template.shape[0]))
    except Exception:
        # fallback: try to proceed without resize if shapes match
        pass

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    score_ssim, _ = ssim(gray_img, gray_template, full=True)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    hist_img = cv2.calcHist([hsv_img],[0,1],None,[50,60],[0,180,0,256])
    hist_template = cv2.calcHist([hsv_template],[0,1],None,[50,60],[0,180,0,256])

    cv2.normalize(hist_img, hist_img, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_template, hist_template, 0, 1, cv2.NORM_MINMAX)

    score_hist = cv2.compareHist(hist_img, hist_template, cv2.HISTCMP_CORREL)
    return 0.85 * score_ssim + 0.15 * score_hist

def identify_ball(pil_crop, ball_templates, threshold=0.7):
    """Return (best_name_or_empty_string, best_score)."""
    icon = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
    best_match, best_score = None, -1
    if not ball_templates:
        return ("", 0.0)
    for name, tmpl in ball_templates.items():
        if tmpl is None:
            # skip missing templates
            continue
        score = compare_ball(icon, tmpl)
        if score > best_score:
            best_match, best_score = name, score
    return (best_match if best_score >= threshold else "", best_score)

def compare_origin_template(img, template):
    """
    Compare origin mark using normalized cross-correlation (cv2.matchTemplate).
    Returns a score between -1 and 1 (we clamp to 0..1 range).
    """
    if img is None or template is None:
        return 0.0

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
    template_gray = cv2.GaussianBlur(template_gray, (3,3), 0)

    # Resize template to input size for single-value match (if necessary)
    if img_gray.shape != template_gray.shape:
        template_gray = cv2.resize(template_gray, (img_gray.shape[1], img_gray.shape[0]))

    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    # after resizing template to same size, res will be a 1x1 array
    try:
        score = float(res[0][0])
    except Exception:
        score = 0.0
    # normalize to 0..1
    return max(0.0, min(1.0, score))

def identify_origin_template(pil_crop, templates, threshold=0.69):
    """Return (best_name_or_empty, best_score)."""
    icon = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
    best_match, best_score = None, -1
    if not templates:
        return ("", 0.0)
    for name, tmpl in templates.items():
        if tmpl is None:
            continue
        score = compare_origin_template(icon, tmpl)
        if score > best_score:
            best_match, best_score = name, score
    return (best_match if best_score >= threshold else "", best_score)


def clean_text_field(s):
    """Strip leading/trailing whitespace and common trailing punctuation."""
    if not s:
        return ""
    return s.strip().strip(string.punctuation)

# -------------------------
# OCR / Processing functions
# -------------------------
def extract_data(pil_img,
                 regions,
                 ball_templates,
                 origin_templates,
                 male_color, female_color, gender_tolerance, gender_sample_pos):
    """
    Extracts all fields from an input PIL image using the provided regions
    and template sets. Returns list of fields matching CSV schema.
    """
    # safety checks
    for required in ("name","nickname","ot","gender","lvl","nature","ability","catchdate","ball","origin"):
        if required not in regions:
            raise KeyError(f"Region '{required}' missing from layout")

    # Crop all regions from the provided layout
    crops = {key: pil_img.crop(box) for key, box in regions.items()}

    # Gender detection (use provided config values)
    gender = checkGenderColor(crops['gender'], male_color=male_color, female_color=female_color,
                              tolerance=gender_tolerance, samplePos=gender_sample_pos)

    # OCR configs
    ocr_config = '--psm 7 --oem 3'  # single line
    ocr_config_digits = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
    ocr_config_date = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/'

    name = pytesseract.image_to_string(crops['name'], config=ocr_config).strip()
    nickname = pytesseract.image_to_string(crops['nickname'], config=ocr_config).strip()
    
    ot_raw = pytesseract.image_to_string(crops['ot'], config=ocr_config)
    ot = clean_text_field(ot_raw).strip()

    lvl_img = preprocess_for_ocr(crops['lvl'])
    lvl = pytesseract.image_to_string(lvl_img, config=ocr_config_digits).strip()
    nature = pytesseract.image_to_string(crops['nature'], config=ocr_config).strip()
    ability = pytesseract.image_to_string(crops['ability'], config=ocr_config).strip()

    date_img = preprocess_for_ocr(crops['catchdate'])
    catchdate_raw = pytesseract.image_to_string(date_img, config=ocr_config_date)
    print(catchdate_raw)
    catchdate = cleanAllButDate(catchdate_raw)

    # Ball detection (uses provided ball_templates)
    ball_name, ball_score = identify_ball(crops['ball'], ball_templates)
    print(f"Ball Type Confidence: {ball_score:.3f} | {ball_name}")

    # Origin detection
    origin_name, origin_score = identify_origin_template(crops['origin'], origin_templates, threshold=0.4)
    print(f"Origin Mark Confidence: {origin_score:.3f} | {origin_name}")

    return [name, nickname, ot, lvl, gender, nature, ability, catchdate, ball_name, origin_name]

def process_queue(queue, csv_file,
                  regions, ball_templates, origin_templates,
                  male_color, female_color, gender_tolerance, gender_sample_pos):
    """Background thread to process OCR without blocking video feed"""
    while True:
        item = queue.get()
        if item is None:  # Poison pill to stop thread
            break
        pil_img = item
        try:
            row = extract_data(pil_img,
                               regions, ball_templates, origin_templates,
                               male_color, female_color, gender_tolerance, gender_sample_pos)
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Captured: {row[0]} | OT: {row[2]} | Lv.{row[3]}")
        except Exception as e:
            print(f"ERROR: failed to process: {e}")
        finally:
            queue.task_done()

# -------------------------
# Main - glue everything
# -------------------------
def main():
    # find configs
    config_files = list_game_configs("configs")
    if not config_files:
        print("ERROR: No configs found in /configs/")
        return

    chosen_config = choose_config(config_files)
    print(f"Loading configuration: {chosen_config}")
    config_filename = os.path.basename(chosen_config)   # "home_box_full.json"
    config_name, _ = os.path.splitext(config_filename)  # correctly unpacks
    print(type(chosen_config), chosen_config)


    main_cfg = load_main_config(chosen_config)

    # load all subconfigs including gender settings
    try:
        (regions,
         ball_templates,
         origin_templates,
         male_color,
         female_color,
         gender_tolerance,
         gender_sample_pos) = load_full_configuration(main_cfg)
    except Exception as e:
        print(f"[FATAL] Failed loading configuration: {e}")
        return

    print("Regions loaded:", list(regions.keys()))
    print("Ball templates:", list(ball_templates.keys()))
    print("Origin templates:", list(origin_templates.keys()))
    print("Gender settings:", male_color, female_color, gender_tolerance, gender_sample_pos)

    # Setup CSV
    outdir = "output"
    os.makedirs(outdir, exist_ok=True)
    csv_file = os.path.join("output", f"{config_name}_pokemon_data.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Pokemon", "Nickname", "OT", "Level", "Gender", "Nature", "Ability", "Catch Date", "Poke Ball", "Origin Mark"])

    # Preview and queue config
    PREVIEW_SCALE = 0.5
    process_queue_obj = Queue(maxsize=20)

    # Start worker thread and pass gender parameters
    worker = Thread(
        target=process_queue,
        args=(process_queue_obj, csv_file,
              regions, ball_templates, origin_templates,
              male_color, female_color, gender_tolerance, gender_sample_pos),
        daemon=True
    )
    worker.start()

    # Video capture init
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF (Windows Media Foundation)"),
        (cv2.CAP_ANY, "Default")
    ]
    cap = None
    for backend, name in backends:
        print(f"Trying {name}...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Using {name}")
                break
            else:
                cap.release()
        else:
            print(f"{name} failed to open")
            cap.release()
    if not cap or not cap.isOpened():
        print("ERROR: Could not open camera with any backend!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except:
        pass

    window_name = "Pokemon Home - OCR Data Extractor Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("="*50)
    print("Pokemon Home - OCR Data Extractor Tool")
    print("="*50)
    print("SPACE - Capture & extract | ESC - Quit | TAB - Settings")
    print("-"*25)
    print("Created by Audrey H.")
    print("="*50)

    last_capture_time = 0
    capture_cooldown = 0.5

    ret, test_frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera")
        cap.release()
        return
    h, w = test_frame.shape[:2]
    preview_size = (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE))

    frame_skip = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.grab()
            continue

        frame_skip += 1
        if frame_skip % 2 != 0:
            continue

        preview = cv2.resize(frame, preview_size, interpolation=cv2.INTER_NEAREST)
        preview_h, preview_w = preview.shape[:2]
        overlay = preview.copy()

        # Draw regions scaled to preview
        for region_name, region_box in regions.items():
            left, top, right, bottom = region_box
            left = int(left * PREVIEW_SCALE)
            top = int(top * PREVIEW_SCALE)
            right = int(right * PREVIEW_SCALE)
            bottom = int(bottom * PREVIEW_SCALE)
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(overlay, region_name, (max(0,left - 20), max(10, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # ---------------------------
        # Right-side Info Box (Space/ESC)
        # ---------------------------
        infobox_w, infobox_h = 250, 90
        infobox_alpha = 0.6
        cv2.rectangle(overlay, (preview_w - infobox_w, 10), (preview_w, 10 + infobox_h + 33), (0, 0, 0), -1)
        cv2.addWeighted(overlay, infobox_alpha, preview, 1 - infobox_alpha, 0, preview)

        # Draw text on preview AFTER blending (fully opaque)
        cv2.putText(preview, "Press [Spacebar] to log,", (preview_w - infobox_w + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(preview, "Press [ESC] to exit.", (preview_w - infobox_w + 10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(preview, "[Tab] for cam settings.", (preview_w - infobox_w + 10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # queue status
        queue_size = process_queue_obj.qsize()
        queue_overlay = preview.copy()
        queue_box_w, queue_box_h = (600, 65) if queue_size > 15 else (250, 65)
        queue_alpha = 1.0 if queue_size >= 20 else (0.8 if queue_size > 15 else 0.6)
        text_color = (0,0,255) if queue_size > 15 else (255,255,255)
        cv2.rectangle(queue_overlay, (10, 10), (10 + queue_box_w, 10 + queue_box_h), (0, 0, 0), -1)
        cv2.addWeighted(queue_overlay, queue_alpha, preview, 1 - queue_alpha, 0, preview)
        box_text = f"Queue: {queue_size}/20"
        if queue_size >= 20:
            box_text += " | ERROR: QUEUE AT MAX"
        elif queue_size > 15:
            box_text += " | WARNING: NEAR QUEUE MAX"
        cv2.putText(preview, box_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        cv2.imshow(window_name, preview)
        cv2.setMouseCallback(window_name, lambda *args: None)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 9:  # TAB
            print("Opening settings dialogue (not implemented)")
        elif key == 32:  # SPACE capture
            current_time = time.time()
            if current_time - last_capture_time < capture_cooldown:
                print("! Too fast! Wait a moment...")
                continue
            if process_queue_obj.full():
                print("! Queue full! Wait for processing to catch up...")
                continue
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            process_queue_obj.put(pil_img)
            print("→ Queued for processing...")
            last_capture_time = current_time

    print("\nShutting down...")
    process_queue_obj.put(None)
    worker.join(timeout=5)
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
