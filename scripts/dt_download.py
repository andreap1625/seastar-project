import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("coralbase").project("coral-detector")
version = project.version(6)
dataset = version.download("yolov8")
                
print("Dataset downloaded successfully!")

