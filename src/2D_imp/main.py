import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

IMAGE_PATH = r"F:\Fac\Licenta\Date\Imagini\test.jpg"
OUTPUT_PATH = r"F:\Fac\Licenta\Date\Output\test_output.png"
WEIGHTS_PATH = r"F:\Fac\Licenta\Models\Weights_sam2\sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

person_detector = YOLO('yolov8n.pt')

def get_main_subject_box(image_bgr):
    """
    Calculeaza care persoana ocupa cel mai mult loc din imagine si o marcheaza ca personajul principal
    Maybe to do - ciclu prin toate persoanele din poza pentru a obtine mai multe modele
    """
    results = person_detector(image_bgr, classes = [0], verbose= False)
    if not results or len(results[0].boxes) == 0:
        print("Nici o persoana nu a fost observata")
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()

    best_box = None
    max_area = 0

    for box in boxes:
        x_min,y_min,x_max,y_max = box
        area = (x_max-x_min) * (y_max-y_min)
        if area > max_area:
            max_area = area
            best_box = np.array([x_min,y_min,x_max,y_max])

    return best_box


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using " + device)

model     = build_sam2(MODEL_CFG, WEIGHTS_PATH, device=device)
predictor = SAM2ImagePredictor(model)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

input_box = get_main_subject_box(image_bgr)

predictor.set_image(image_rgb)

h, w = image_rgb.shape[:2]
cx, cy = w // 2, h // 2

point_coords = np.array([[cx, cy]])
point_labels = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box = input_box[None, :],
    multimask_output=True
)

best_idx = np.argmax(scores)
best_mask = masks[best_idx]


mask_uint8 =  (best_mask * 255).astype(np.uint8)

image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
image_rgba[:, :, 3] = mask_uint8

cv2.imwrite(OUTPUT_PATH, image_rgba)


