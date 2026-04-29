import os
import torch
import torch.nn as nn
os.environ.pop("SSL_CERT_FILE", None)
import timm
import gradio as gr
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import tempfile


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
YOLO_WEIGHTS   = "runs/detect/runs/yolo-coral-detector/weights/best.pt"
EFFNET_WEIGHTS = "models/best_efficientnet.pth"
CLASSES        = ['bleached_corals', 'healthy_corals']
CONF_THRESHOLD = 0.25
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE       = 224

COLORS = {
    'bleached_corals': (255, 80,  60),   # rojo vibrante (RGB)
    'healthy_corals':  (60,  220, 120),  # verde vibrante (RGB)
}

# ══════════════════════════════════════════════════════════════
#  CARGA DE MODELOS (se hace una sola vez)
# ══════════════════════════════════════════════════════════════
_yolo    = None
_effnet  = None
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_models():
    global _yolo, _effnet
    if _yolo is not None and _effnet is not None:
        return True, "Modelos ya cargados ✓"

    errors = []

    # YOLO
    try:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_WEIGHTS)
    except Exception as e:
        errors.append(f"YOLO: {e}")

    # EfficientNet
    try:
        backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        backbone.classifier = nn.Identity()

        head = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2)
        )

        class CoralClassifier(nn.Module):
            def __init__(self, b, h):
                super().__init__()
                self.backbone = b
                self.head     = h
            def forward(self, x):
                return self.head(self.backbone(x))

        model = CoralClassifier(backbone, head).to(DEVICE)
        model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=DEVICE))
        model.eval()
        _effnet = model
    except Exception as e:
        errors.append(f"EfficientNet: {e}")

    if errors:
        return False, "Errores al cargar modelos:\n" + "\n".join(errors)
    return True, f"Modelos cargados correctamente en {DEVICE} ✓"


# ══════════════════════════════════════════════════════════════
#  INFERENCIA EN UN FRAME (numpy BGR → numpy RGB anotado)
# ══════════════════════════════════════════════════════════════
def process_frame(frame_bgr):
    """Recibe frame BGR (numpy), devuelve frame RGB anotado + stats."""
    results = _yolo(frame_bgr, conf=CONF_THRESHOLD, verbose=False)[0]

    stats = {"total": 0, "bleached": 0, "healthy": 0, "detections": []}
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    draw      = ImageDraw.Draw(pil_img, "RGBA")

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_yolo       = float(box.conf[0])

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor   = _transform(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits    = _effnet(tensor)
            probs     = torch.softmax(logits, dim=1)
            conf_eff  = float(probs.max())
            label_idx = int(probs.argmax())
            label     = CLASSES[label_idx]

        color = COLORS[label]
        r, g, b = color

        # Bounding box con relleno semitransparente
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=3)
        draw.rectangle([x1, y1, x2, y1 + 28], fill=(r, g, b, 180))

        # Texto
        txt = f"{label.replace('_', ' ')}  {conf_eff:.0%}"
        draw.text((x1 + 6, y1 + 5), txt, fill=(255, 255, 255, 255))

        stats["total"] += 1
        if label == "bleached_corals":
            stats["bleached"] += 1
        else:
            stats["healthy"] += 1

        stats["detections"].append({
            "label": label,
            "conf_effnet": round(conf_eff, 3),
            "conf_yolo":   round(conf_yolo, 3),
            "bbox": [x1, y1, x2, y2]
        })

    return np.array(pil_img), stats


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN PARA IMAGEN
# ══════════════════════════════════════════════════════════════
def run_on_image(image, conf_thresh):
    global CONF_THRESHOLD
    CONF_THRESHOLD = conf_thresh

    ok, msg = load_models()
    if not ok:
        return None, msg, ""

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, stats = process_frame(frame_bgr)

    # Resumen en markdown
    pct_bleached = (stats["bleached"] / stats["total"] * 100) if stats["total"] else 0
    pct_healthy  = (stats["healthy"]  / stats["total"] * 100) if stats["total"] else 0

    summary = f"""### Resultados de detección

| Métrica | Valor |
|---|---|
| **Corales detectados** | {stats["total"]} |
| 🔴 Blanqueados | {stats["bleached"]} ({pct_bleached:.0f}%) |
| 🟢 Saludables | {stats["healthy"]} ({pct_healthy:.0f}%) |
| **Dispositivo** | {DEVICE} |

"""
    if stats["detections"]:
        summary += "#### Detalle por coral\n"
        for i, d in enumerate(stats["detections"], 1):
            emoji = "🔴" if d["label"] == "bleached_corals" else "🟢"
            summary += (f"**#{i}** {emoji} `{d['label'].replace('_',' ')}` — "
                        f"Clasificador: **{d['conf_effnet']:.0%}** | "
                        f"Detector: {d['conf_yolo']:.0%}\n\n")

    return Image.fromarray(annotated), msg, summary


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN PARA VIDEO
# ══════════════════════════════════════════════════════════════
def run_on_video(video_path, conf_thresh, progress=gr.Progress()):
    global CONF_THRESHOLD
    CONF_THRESHOLD = conf_thresh

    ok, msg = load_models()
    if not ok:
        return None, msg, ""

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    total_stats = {"total": 0, "bleached": 0, "healthy": 0}
    frame_idx   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_rgb, stats = process_frame(frame)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        writer.write(annotated_bgr)

        total_stats["total"]    += stats["total"]
        total_stats["bleached"] += stats["bleached"]
        total_stats["healthy"]  += stats["healthy"]

        frame_idx += 1
        if total_frames > 0:
            progress(frame_idx / total_frames, desc=f"Procesando frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    pct_b = (total_stats["bleached"] / total_stats["total"] * 100) if total_stats["total"] else 0
    pct_h = (total_stats["healthy"]  / total_stats["total"] * 100) if total_stats["total"] else 0

    summary = f"""### Resumen del video

| Métrica | Valor |
|---|---|
| **Frames procesados** | {frame_idx} |
| **Detecciones totales** | {total_stats["total"]} |
| 🔴 Blanqueados | {total_stats["bleached"]} ({pct_b:.0f}%) |
| 🟢 Saludables | {total_stats["healthy"]} ({pct_h:.0f}%) |
| **Dispositivo** | {DEVICE} |
"""
    return out_path, msg, summary


# ══════════════════════════════════════════════════════════════
#  INTERFAZ GRADIO
# ══════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --ocean-dark:   #040d1a;
    --ocean-mid:    #071e38;
    --ocean-light:  #0a3055;
    --accent-teal:  #00c8b4;
    --accent-coral: #ff5f52;
    --accent-green: #3ddc84;
    --text-main:    #e8f4f8;
    --text-muted:   #7ab3cc;
    --border:       rgba(0,200,180,0.18);
}

body, .gradio-container {
    background: var(--ocean-dark) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text-main) !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    background: linear-gradient(180deg, rgba(0,200,180,0.06) 0%, transparent 100%);
}
.app-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.4rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent-teal) 30%, #5bf5e8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.app-header p {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Tabs */
.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.7rem 1.4rem !important;
}
.tab-nav button.selected {
    color: var(--accent-teal) !important;
    border-bottom-color: var(--accent-teal) !important;
}

/* Panels */
.panel {
    background: var(--ocean-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.4rem !important;
}

/* Buttons */
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent-teal), #00a896) !important;
    color: var(--ocean-dark) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(0,200,180,0.25) !important;
}
button.primary-btn:hover, .gr-button-primary:hover {
    box-shadow: 0 0 30px rgba(0,200,180,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Sliders & inputs */
.gr-slider input[type=range]::-webkit-slider-thumb { background: var(--accent-teal) !important; }
.gr-slider input[type=range]::-webkit-slider-runnable-track { background: var(--ocean-light) !important; }

/* Legend badges */
.legend { display: flex; gap: 1.5rem; justify-content: center; margin: 1rem 0; }
.badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    padding: 6px 14px; border-radius: 20px; font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-red  { background: rgba(255,95,82,0.15); color: #ff7a70; border: 1px solid rgba(255,95,82,0.35); }
.badge-green{ background: rgba(61,220,132,0.15); color: #3ddc84; border: 1px solid rgba(61,220,132,0.35); }
.dot { width: 9px; height: 9px; border-radius: 50%; }
.dot-red  { background: #ff5f52; box-shadow: 0 0 8px #ff5f52; }
.dot-green{ background: #3ddc84; box-shadow: 0 0 8px #3ddc84; }

/* Status box */
.status-box {
    background: rgba(0,200,180,0.05);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* Markdown output */
.gr-markdown { color: var(--text-main) !important; }
.gr-markdown table { border-collapse: collapse; width: 100%; }
.gr-markdown th { 
    background: rgba(0,200,180,0.12) !important; 
    color: var(--accent-teal) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    padding: 8px 14px;
    text-align: left;
}
.gr-markdown td { 
    border-top: 1px solid var(--border); 
    padding: 8px 14px; 
    font-size: 0.88rem;
}
"""

LEGEND_HTML = """
<div class="legend">
  <span class="badge badge-red"><span class="dot dot-red"></span>BLEACHED CORAL</span>
  <span class="badge badge-green"><span class="dot dot-green"></span>HEALTHY CORAL</span>
</div>
"""

HEADER_HTML = """
<div class="app-header">
  <h1>🪸 SEASTAR CORAL PIPELINE</h1>
  <p>YOLOv8 detection · EfficientNet-B0 classification · Real-time reef health monitoring</p>
</div>
"""

with gr.Blocks(css=CSS, title="SEASTAR — Coral Pipeline Demo") as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(LEGEND_HTML)

    with gr.Tabs():

        # ── TAB 1: Imagen ──────────────────────────────────────
        with gr.TabItem("📷  Imagen"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    img_input = gr.Image(
                        label="Imagen de arrecife",
                        type="pil",
                        height=320,
                        sources=["upload", "clipboard"]
                    )
                    conf_img = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Umbral de confianza YOLO"
                    )
                    btn_img = gr.Button("▶  Analizar imagen", variant="primary")

                with gr.Column(scale=1, elem_classes="panel"):
                    img_output = gr.Image(label="Resultado anotado", height=320)

            with gr.Row():
                status_img  = gr.Textbox(label="Estado del sistema", lines=1, elem_classes="status-box")
                summary_img = gr.Markdown(label="Reporte")

            btn_img.click(
                fn=run_on_image,
                inputs=[img_input, conf_img],
                outputs=[img_output, status_img, summary_img]
            )

        # ── TAB 2: Video ───────────────────────────────────────
        with gr.TabItem("🎬  Video"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes="panel"):
                    vid_input = gr.Video(
                        label="Video de arrecife",
                        height=300,
                        sources=["upload"]
                    )
                    conf_vid = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="Umbral de confianza YOLO"
                    )
                    btn_vid = gr.Button("▶  Procesar video", variant="primary")
                    gr.Markdown(
                        "_El video se procesará frame a frame. Para videos largos esto puede tomar varios minutos._",
                    )

                with gr.Column(scale=1, elem_classes="panel"):
                    vid_output = gr.Video(label="Video anotado", height=300)

            with gr.Row():
                status_vid  = gr.Textbox(label="Estado del sistema", lines=1, elem_classes="status-box")
                summary_vid = gr.Markdown(label="Reporte")

            btn_vid.click(
                fn=run_on_video,
                inputs=[vid_input, conf_vid],
                outputs=[vid_output, status_vid, summary_vid]
            )

#         # ── TAB 3: Info ────────────────────────────────────────
#         with gr.TabItem("ℹ️  Pipeline"):
#             gr.Markdown("""
# ## Arquitectura del pipeline

# ```
# Entrada (imagen / frame de video)
#         │
#         ▼
#  ┌─────────────┐
#  │  YOLOv8     │  ← Detecta y localiza corales
#  │  Detector   │     Genera bounding boxes
#  └──────┬──────┘
#         │  crops
#         ▼
#  ┌─────────────────────┐
#  │  EfficientNet-B0    │  ← Clasifica cada crop
#  │  + MLP head         │     bleached / healthy
#  └──────┬──────────────┘
#         │  label + confidence
#         ▼
#  Anotación visual + reporte estadístico
# ```

# ### Clases
# | Clase | Color | Descripción |
# |---|---|---|
# | `bleached_corals` | 🔴 Rojo | Coral blanqueado (estrés térmico) |
# | `healthy_corals`  | 🟢 Verde | Coral en buen estado |

# ### Modelos
# - **Detector**: YOLOv8 custom entrenado sobre dataset de arrecifes
# - **Clasificador**: EfficientNet-B0 con cabeza MLP (Linear → BN → ReLU → Dropout → Linear)
# - **Input del clasificador**: 224×224 px, normalizado ImageNet

# ### Parámetros
# - `CONF_THRESHOLD`: umbral mínimo de confianza del detector (ajustable con el slider)
# - `IMG_SIZE`: 224 × 224
# - `DEVICE`: CUDA si disponible, si no CPU
# """)

#     gr.HTML("""
#     <div style="text-align:center; padding: 1.5rem 0 0.5rem;
#          font-family:'Space Mono',monospace; font-size:0.7rem;
#          color:rgba(122,179,204,0.4); letter-spacing:1px;">
#       SEASTAR CORAL PIPELINE DEMO · YOLO + EFFICIENTNET-B0
#     </div>
#     """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)