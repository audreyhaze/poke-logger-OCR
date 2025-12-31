import cv2
import string
import json
import glob
import os
import re
import csv
import time
import numpy as np
import mss
import pygetwindow as gw
import pytesseract
from PIL import Image
from threading import Thread, Lock
from queue import Queue
from skimage.metrics import structural_similarity as ssim

# Path to your tesseract exe - Update this if yours is in a different location
pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------
# 1. VIDEO SOURCE SELECTION (Camera or Window)
# ---------------------------------------------------------
def choose_source():
    print("\n" + "="*30)
    print("  POKEMON DATA EXTRACTOR  ")
    print("="*30)
    print("[1] Camera / Capture Card")
    print("[2] Window Capture (Emulator/Stream)")
    
    choice = input("\nSelect Source Type: ").strip()
    
    if choice == "1":
        # Scan for cameras
        available = []
        print("Scanning for devices...")
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret: available.append(i)
                cap.release()
        
        if not available:
            print("No cameras found. Defaulting to Index 0.")
            return "camera", 0
        
        for i, idx in enumerate(available):
            print(f"[{i}] Camera Index {idx}")
        return "camera", available[int(input("Select Camera Index: "))]
    
    else:
        # Scan for windows
        titles = [t for t in gw.getAllTitles() if t.strip()]
        for i, t in enumerate(titles):
            print(f"[{i}] {t}")
        return "window", titles[int(input("Select Window Index: "))]

# ---------------------------------------------------------
# 2. UNIVERSAL STREAM CLASS (High Speed / No Lag)
# ---------------------------------------------------------
class UniversalStream:
    def __init__(self, source_type, target):
        self.source_type = source_type
        self.target = target
        self.frame = None
        self.grabbed = False
        self.read_lock = Lock()
        self.stopped = False

        if self.source_type == "camera":
            self.cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.sct = mss.mss()
            try:
                self.window = gw.getWindowsWithTitle(target)[0]
            except IndexError:
                print(f"Error: Could not find window '{target}'")
                exit()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.source_type == "camera":
                grabbed, frame = self.cap.read()
            else:
                # Capture specific window coords
                rect = {"top": self.window.top, "left": self.window.left, 
                        "width": self.window.width, "height": self.window.height}
                sct_img = self.sct.grab(rect)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                grabbed = True

            if grabbed:
                with self.read_lock:
                    self.frame = frame
                    self.grabbed = grabbed

    def get_frame(self):
        with self.read_lock:
            return self.frame.copy() if self.grabbed and self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.source_type == "camera":
            self.cap.release()

# ---------------------------------------------------------
# 3. OCR & IMAGE LOGIC
# ---------------------------------------------------------
def preprocess_for_ocr(crop_img):
    gray = crop_img.convert('L')
    return gray.point(lambda x: 0 if x < 128 else 255, '1')

def check_gender(img, male_c, female_c, tol, pos):
    try:
        pixel = img.getpixel(pos)
        if all(abs(int(a)-int(b)) <= tol for a,b in zip(pixel, male_c)): return "M"
        if all(abs(int(a)-int(b)) <= tol for a,b in zip(pixel, female_c)): return "F"
    except: pass
    return ""

def identify_template(crop, templates, threshold=0.6):
    if not templates: return ""
    cv_img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    best_name, best_score = "", -1
    for name, tmpl in templates.items():
        if tmpl is None: continue
        # Resize template to match crop size for SSIM
        t_resized = cv2.resize(tmpl, (cv_img.shape[1], cv_img.shape[0]))
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray_tmpl = cv2.cvtColor(t_resized, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_img, gray_tmpl, full=True)
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= threshold else ""

def ocr_worker(queue, csv_path, regions, ball_tmpls, origin_tmpls, m_c, f_c, tol, pos):
    while True:
        pil_img = queue.get()
        if pil_img is None: break
        try:
            crops = {k: pil_img.crop(v) for k, v in regions.items()}
            
            # Text Fields
            name = pytesseract.image_to_string(crops['name'], config='--psm 7').strip()
            ot = pytesseract.image_to_string(crops['ot'], config='--psm 7').strip()
            lvl = pytesseract.image_to_string(preprocess_for_ocr(crops['lvl']), 
                                              config='--psm 7 -c tessedit_char_whitelist=0123456789').strip()
            
            # Template/Color Fields
            gender = check_gender(crops['gender'], m_c, f_c, tol, pos)
            ball = identify_template(crops['ball'], ball_tmpls)
            origin = identify_template(crops['origin'], origin_tmpls)

            row = [name, "", ot, lvl, gender, "", "", "", ball, origin]
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            print(f"> LOGGED: {name} (Lv.{lvl}) | OT: {ot} | Ball: {ball}")
        except Exception as e:
            print(f"OCR Error: {e}")
        finally:
            queue.task_done()

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    # Source & Config Setup
    src_type, target = choose_source()
    
    config_files = [os.path.join("configs", f) for f in os.listdir("configs") if f.endswith(".json")]
    if not config_files:
        print("No JSON configs found in /configs/")
        return
    
    print("\nSelect Configuration:")
    for i, f in enumerate(config_files): print(f"[{i}] {f}")
    main_cfg = json.load(open(config_files[int(input("Select Index: "))]))

    # Load Layout & Templates
    layout_cfg = json.load(open(main_cfg["layout"]))
    regions = {k: (v[0], v[1], v[0]+v[2], v[1]+v[3]) for k, v in layout_cfg.items()}
    
    def load_tmpls(path):
        res = {}
        for f in glob.glob(os.path.join(path, "*")):
            if f.lower().endswith((".png", ".jpg")):
                res[os.path.splitext(os.path.basename(f))[0]] = cv2.imread(f)
        return res

    ball_tmpls = load_tmpls(main_cfg["ball"])
    origin_tmpls = load_tmpls(main_cfg["origin"])

    # UI & CSV Setup
    os.makedirs("output", exist_ok=True)
    csv_path = os.path.join("output", "pokemon_data.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Pokemon", "Nickname", "OT", "Level", "Gender", "Nature", "Ability", "Date", "Ball", "Origin"])

    # Launch Threads
    pq = Queue(maxsize=20)
    m_c = tuple(int(x) for x in main_cfg.get("male_color", "100,128,255").split(","))
    f_c = tuple(int(x) for x in main_cfg.get("female_color", "255,77,62").split(","))
    tol = int(main_cfg.get("gender_tolerance", 30))
    pos = tuple(int(x) for x in main_cfg.get("gender_sample_pos", "12,22").split(","))

    Thread(target=ocr_worker, args=(pq, csv_path, regions, ball_tmpls, origin_tmpls, m_c, f_c, tol, pos), daemon=True).start()
    stream = UniversalStream(src_type, target).start()

    cv2.namedWindow("OCR Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OCR Preview", 1280, 720)

    print("\n" + "="*30)
    print("LIVE STREAM ACTIVE")
    print("SPACE : Capture & Log")
    print("ESC   : Quit")
    print("="*30)

    while True:
        frame = stream.get_frame()
        if frame is None: continue

        # Display Frame with Overlays
        preview = cv2.resize(frame, (1280, 720))
        scale_x = 1280 / frame.shape[1]
        scale_y = 720 / frame.shape[0]

        for name, box in regions.items():
            cv2.rectangle(preview, (int(box[0]*scale_x), int(box[1]*scale_y)), 
                          (int(box[2]*scale_x), int(box[3]*scale_y)), (0, 255, 0), 1)

        cv2.putText(preview, f"Queue: {pq.qsize()}", (20, 50), 1, 2, (0, 255, 0), 2)
        cv2.imshow("OCR Preview", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == 32: # SPACE
            if not pq.full():
                pq.put(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                print(">>> Captured. Processing...")

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()