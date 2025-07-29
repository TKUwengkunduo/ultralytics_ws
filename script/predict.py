from ultralytics import YOLO
import os

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        初始化 YOLO 模型
        :param model_path: 模型檔路徑 (如: 'yolov8n.pt')
        """
        if not os.path.exists(model_path) and not model_path.startswith("yolov8"):
            raise FileNotFoundError(f"找不到指定的模型檔案: {model_path}")
        self.model = YOLO(model_path)

    def predict(self, source, conf=0.25, iou=0.45, save=True, show=False, device=None, classes=None, imgsz=640):
        """
        使用 YOLO 模型進行檢測
        :param source: 檢測來源 (圖片路徑, 影片路徑, 0=webcam)
        :param conf: 置信度閾值 (0-1)
        :param iou: IoU 閾值 (0-1)
        :param save: 是否儲存結果
        :param show: 是否顯示結果
        :param device: 運行裝置 ('cpu' 或 '0' 表示 GPU)
        :param classes: 只檢測特定類別 (list of int)
        :param imgsz: 輸入影像大小
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            show=show,
            device=device,
            classes=classes,
            imgsz=imgsz
        )
        return results


def main():
    # === 這裡設定所有參數 ===
    CONFIG = {
        "model_path": "runs/detect/renesas/weights/best.pt",          # 模型檔路徑
        "source": 0,                # 檢測來源 (圖片, 影片, 或 0=Webcam)
        "conf": 0.8,                         # 置信度閾值
        "iou": 0.5,                          # IoU 閾值
        "save": False,                        # 是否儲存結果
        "show": True,                       # 是否顯示結果
        "device": "0",                       # 使用 GPU: "0" 或 CPU: "cpu"
        "classes": None,                      # 只檢測特定類別 (如 0=person)，None=全部
        "imgsz": 640                         # 輸入圖片大小
    }

    # 建立檢測器
    detector = YOLODetector(CONFIG["model_path"])

    # 執行檢測
    results = detector.predict(
        source=CONFIG["source"],
        conf=CONFIG["conf"],
        iou=CONFIG["iou"],
        save=CONFIG["save"],
        show=CONFIG["show"],
        device=CONFIG["device"],
        classes=CONFIG["classes"],
        imgsz=CONFIG["imgsz"]
    )

    # 顯示結果摘要
    print("檢測完成! 結果摘要：")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
