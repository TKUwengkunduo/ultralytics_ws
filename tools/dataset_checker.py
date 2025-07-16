import os
from typing import List

class DatasetChecker:
    def __init__(self, dataset_dir: str, allowed_class_ids: List[int]):
        self.dataset_dir = dataset_dir
        self.allowed_class_ids = allowed_class_ids
        self.supported_image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    def _get_image_files(self):
        return [f for f in os.listdir(self.dataset_dir)
                if os.path.splitext(f)[1].lower() in self.supported_image_exts]

    def _get_label_files(self):
        return [f for f in os.listdir(self.dataset_dir) if f.endswith('.txt')]

    def _has_only_allowed_classes(self, label_path: str) -> bool:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                if class_id not in self.allowed_class_ids:
                    return False
            except ValueError:
                return False
        return True

    def clean_dataset(self):
        image_files = self._get_image_files()
        label_files = self._get_label_files()

        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}

        # 檢查：圖片沒有對應標註
        for image_file in image_files:
            basename, ext = os.path.splitext(image_file)
            if basename not in label_basenames:
                os.remove(os.path.join(self.dataset_dir, image_file))
                print(f"Removed image without label: {image_file}")

        # 檢查：標註檔沒有對應圖片，或含有非法類別
        for label_file in label_files:
            basename = os.path.splitext(label_file)[0]
            label_path = os.path.join(self.dataset_dir, label_file)

            # 找對應圖片
            image_path = None
            for ext in self.supported_image_exts:
                possible_image = os.path.join(self.dataset_dir, basename + ext)
                if os.path.exists(possible_image):
                    image_path = possible_image
                    break

            if not image_path:
                os.remove(label_path)
                print(f"Removed label without image: {label_file}")
                continue

            # 檢查標註檔中的類別是否合法
            if not self._has_only_allowed_classes(label_path):
                os.remove(label_path)
                os.remove(image_path)
                print(f"Removed pair with invalid class: {label_file}, {os.path.basename(image_path)}")

        print("Dataset check and cleaning completed.")

# ------------------------------
# ⚙️ 使用範例配置
# ------------------------------
if __name__ == "__main__":
    dataset_path = "datasets/human_dataset/coco_dataset"  # 你的資料集資料夾
    allowed_classes = [0]  # 只允許 class 0

    checker = DatasetChecker(dataset_dir=dataset_path, allowed_class_ids=allowed_classes)
    checker.clean_dataset()
