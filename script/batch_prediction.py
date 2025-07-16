# yolofolder_predictor.py
import os
from ultralytics import YOLO

class FolderPredictor:
    def __init__(self,
                 model_path: str = "yolo12l.pt",
                 source_folder: str = "./data",
                 save_results: bool = False,
                 save_crops: bool = False,
                 img_size: int = 640,
                 conf: float = 0.25,
                 iou: float = 0.7,
                 device: str = None):
        """
        Args:
            model_path: path to YOLO model (.pt or .yaml)
            source_folder: folder path (supports recursive search)
            save_results: whether to save annotated output
            save_crops: whether to save cropped detections
            img_size: resize size for inference
            conf: confidence threshold
            iou: NMS IoU threshold
            device: 'cpu', 'cuda', or specific GPU index
        """
        self.model = YOLO(model_path)
        self.source_folder = source_folder
        self.save = save_results
        self.save_crop = save_crops
        self.imgsz = img_size
        self.conf = conf
        self.iou = iou
        self.device = device

    def run(self):
        source_pattern = os.path.join(self.source_folder, "**", "*")
        results = self.model.predict(source=source_pattern,
                                     stream=True,
                                     save=self.save,
                                     save_crop=self.save_crop,
                                     imgsz=self.imgsz,
                                     conf=self.conf,
                                     iou=self.iou,
                                     device=self.device,
                                     project="runs/predict",
                                     exist_ok=True)
        for r in results:
            print(f"[INFO] Processed: {r.path} -> {len(r.boxes)} boxes detected")
        print("[INFO] Inference complete.")

if __name__ == "__main__":
    def example1():
        print("\nExample 1: Image prediction")
        predictor = FolderPredictor(model_path='yolo12x.pt',
                                    source_folder="datasets/test_images",
                                    save_results=True,
                                    save_crops=False,
                                    img_size=640,
                                    conf=0.4,
                                    device=0)
        predictor.run()

    def example2():
        print("\nExample 2: Video prediction")
        predictor = FolderPredictor(model_path="yolo11s.pt",
                                    source_folder="./videos",
                                    save_results=True,
                                    save_crops=True,
                                    img_size=640,
                                    conf=0.4,
                                    device=0)
        predictor.run()

    example1()
    # example2()
