import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# ── Configuración ──────────────────────────────────────
YOLO_WEIGHTS   = "runs\\detect\\runs\\yolo-coral-detector\\weights\\best.pt"
EFFNET_WEIGHTS = "models\\best_efficientnet.pth"
CLASSES        = ['bleached_corals', 'healthy_corals']
CONF_THRESHOLD = 0.25
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE       = 224

# ── Cargar YOLO ────────────────────────────────────────
yolo = YOLO(YOLO_WEIGHTS)

# ── Cargar EfficientNet ────────────────────────────────
backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
backbone.classifier = nn.Identity()

features_dim = 1280

classification_head = nn.Sequential(
    nn.Linear(features_dim, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 2)
)

class CoralClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

effnet = CoralClassifier(backbone, classification_head).to(DEVICE)
effnet.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=DEVICE))
effnet.eval()

# ── Transform para EfficientNet ────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Colores por clase ──────────────────────────────────
COLORS = {
    'bleached_corals': (0, 0, 255),  # rojo
    'healthy_corals':  (0, 255, 0)     # verde
}

# ── Pipeline ───────────────────────────────────────────
def process_frame(frame):
    results = yolo(frame, conf=CONF_THRESHOLD, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_yolo       = float(box.conf[0])

        # Crop del coral detectado
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Pasar crop a EfficientNet
        crop_pil    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(crop_pil)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits      = effnet(tensor)
            probs       = torch.softmax(logits, dim=1)
            conf_effnet = float(probs.max())
            label_idx   = int(probs.argmax())
            label       = CLASSES[label_idx]

        # Dibujar bounding box y etiqueta
        color = COLORS[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf_effnet:.2f} | coral {conf_yolo:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ── Inferencia sobre video ─────────────────────────────
if __name__ == '__main__':
    SOURCE = "test_samples/coral_reef_v.mp4"  # cambia esto o pon 0 para webcam
    cap    = cv2.VideoCapture(SOURCE)

    # Guardar video de salida
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("results/output_pipeline_v.mp4",
                          int(cv2.VideoWriter.fourcc(*'mp4v')),
                          fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        out.write(frame)
        # cv2.imshow("SEASTAR Pipeline", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print("Video guardado en output_pipeline.mp4")