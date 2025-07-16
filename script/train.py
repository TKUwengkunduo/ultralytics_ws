from ultralytics import YOLO

class YOLOTrainer:
    def __init__(
        self,
        #======================== Train Settings ========================#
        model: str = None,                      # None
        data: str = None,                       # None
        epochs: int = 100,                      # 100
        time: float = None,                     # None
        patience: int = 100,                    # 100
        batch: float = 8,                       # 16
        imgsz: int = 640,                       # 640
        save: bool = True,                      # True
        save_period: int = 1,                   # -1
        cache: bool = False,                    # False
        device = 0,                             # None ## [0,1]
        workers: int = 32,                      # 8
        project: str = None,                    # None
        name: str = "human0716",                # None
        exist_ok: bool = False,                 # False
        pretrained = True,                      # True
        optimizer: str = 'auto',                # auto
        fraction: float = 1.0,                  # 1.0
        profile: bool = False,                  # False
        freeze = None,                          # None
        lr0: float = 0.01,                      # 0.01
        lrf: float = 0.01,                      # 0.01
        momentum: float = 0.937,                # 0.937
        weight_decay: float = 0.0005,           # 0.0005
        #===================== Augmentation Settings =====================#
        hsv_h: float = 0.015,                   # 0.015
        hsv_s: float = 0.7,                     # 0.7
        hsv_v: float = 0.4,                     # 0.4
        degrees: float = 0.0,                   # 0.0
        translate: float = 0.1,                 # 0.1
        scale: float = 0.5,                     # 0.5
        shear: float = 10.0,                    # 0.0
        perspective: float = 0.0005,            # 0.0
        flipud: float = 0.0,                    # 0.0
        mosaic: float = 1.0,                    # 1.0
        mixup: float = 0.0,                     # 0.0
        cutmix: float = 0.0,                    # 0.0
        auto_augment: str = 'randaugment',      # randaugment
        erasing: float = 0.4,                   # 0.4
    ):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.time = time
        self.patience = patience
        self.batch = batch
        self.imgsz = imgsz
        self.save = save
        self.save_period = save_period
        self.cache = cache
        self.device = device
        self.workers = workers
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.fraction = fraction
        self.profile = profile
        self.freeze = freeze
        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.mosaic = mosaic
        self.cutmix = cutmix
        self.mixup = mixup
        self.auto_augment = auto_augment
        self.erasing = erasing

    def train(self):
        assert self.model is not None, "請指定 model 權重或 yaml 路徑"
        assert self.data is not None, "請指定 data yaml 路徑"
        model = YOLO(self.model)
        # 呼叫 train，官方參數預設如下 :contentReference[oaicite:1]{index=1}
        return model.train(
            data=self.data,
            epochs=self.epochs,
            time=self.time,
            patience=self.patience,
            batch=self.batch,
            imgsz=self.imgsz,
            save=self.save,
            save_period=self.save_period,
            cache=self.cache,
            device=self.device,
            workers=self.workers,
            project=self.project,
            name=self.name,
            exist_ok=self.exist_ok,
            pretrained=self.pretrained,
            optimizer=self.optimizer,
            fraction=self.fraction,
            profile=self.profile,
            freeze=self.freeze,
            lr0=self.lr0,
            lrf=self.lrf,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            hsv_h=self.hsv_h,
            hsv_s=self.hsv_s,
            hsv_v=self.hsv_v,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            flipud=self.flipud,
            mosaic=self.mosaic,
            cutmix=self.cutmix,
            mixup=self.mixup,
            auto_augment=self.auto_augment,
            erasing=self.erasing,
        )


# 使用範例
if __name__ == "__main__":
    trainer = YOLOTrainer(
        model="yolo12x.pt",
        data="datasets/human_dataset/human.yaml"
    )

    results = trainer.train()
    print(results)
