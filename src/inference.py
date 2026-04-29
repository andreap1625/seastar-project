from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("C:\\Users\\andre\\seastar-project\\runs\\detect\\runs\\yolo-coral-detector\\weights\\best.pt")

    results = model.predict(
        source="coral_reef_yt.mp4", 
        save=True,              
        conf=0.25,              
        project="runs",
        name="seastar_inference"
    )

