from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    results = model.train(
        data='Coral-Detector-6/data.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        name='yolo-coral-detector',
        project="runs",
        device=0,
        patience=10, 
        save=True,
        plots=True)
    
    # Validación
    metrics = model.val()
    
    print("\n--- MÉTRICAS DE VALIDACIÓN ---")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precisión:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")