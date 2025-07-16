"""========================================================================================================
#
# ==============================================
# DatasetPreparer is a Python class designed for preprocessing image datasets.
# Its main capabilities include:
# 1. Scanning a specified dataset directory to find image files with given extensions (e.g., .jpg, .png),
#    and checking whether corresponding annotation files (e.g., .txt) exist.
# 2. Randomly splitting valid image-annotation pairs into training, validation, and test sets
#    based on user-defined ratios.
# 3. Writing the results into three files: train.txt, val.txt, and test.txt,
#    each listing the full paths of images in the corresponding set.
#
# Use Case:
# This tool is especially helpful for preparing datasets for object detection models such as YOLO,
# where automatic dataset organization is essential.
# ==============================================
# 
# # ==============================================
# DatasetPreparer 是一個用於影像資料集前處理的 Python 類別。
# 它的主要功能包括：
# 1. 掃描指定資料夾下所有符合影像副檔名（如 .jpg, .png）的檔案，
#    並確認是否有對應的標註檔案（如 .txt）。
# 2. 將符合條件的影像-標註對，依據訓練、驗證、測試的比例（可自訂）進行隨機分割。
# 3. 將分割結果分別寫入 train.txt、val.txt、test.txt 檔案中，每行為影像檔案的完整路徑。
#
# 使用情境：
# 適用於如 YOLO 等物件偵測模型訓練前的資料準備階段，能夠自動掃描資料並分割成訓練集、驗證集與測試集。
# ==============================================
#
========================================================================================================"""

import os
import random
from typing import List, Tuple

class DatasetPreparer:
    def __init__(
        self,
        dataset_root: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
        annotation_extension: str = '.txt',
        seed: int = 42
    ):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.image_extensions = image_extensions
        self.annotation_extension = annotation_extension
        self.seed = seed
        self.image_label_pairs: List[str] = []

        self._validate_ratios()
        random.seed(self.seed)

    def _validate_ratios(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

    def collect_valid_pairs(self):
        print(f"Scanning '{self.dataset_root}' for image-label pairs...")
        for root, _, files in os.walk(self.dataset_root):
            for file in files:
                if file.lower().endswith(self.image_extensions):
                    image_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    annotation_path = os.path.join(root, base_name + self.annotation_extension)
                    if os.path.exists(annotation_path):
                        self.image_label_pairs.append(image_path)
                    else:
                        print(f"Warning: Missing annotation for image '{image_path}'")
        print(f"Total valid image-label pairs found: {len(self.image_label_pairs)}")

    def split_dataset(self):
        print("Splitting dataset into train/val/test...")
        random.shuffle(self.image_label_pairs)
        total = len(self.image_label_pairs)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_set = self.image_label_pairs[:train_end]
        val_set = self.image_label_pairs[train_end:val_end]
        test_set = self.image_label_pairs[val_end:]

        print(f"Train set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")
        print(f"Test set size: {len(test_set)}")

        return train_set, val_set, test_set

    def write_split_files(self, train_set: List[str], val_set: List[str], test_set: List[str]):
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_file('train.txt', train_set)
        self._write_file('val.txt', val_set)
        self._write_file('test.txt', test_set)
        print(f"Split files written to '{self.output_dir}'")

    def _write_file(self, filename: str, dataset: List[str]):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            for img_path in dataset:
                f.write(f"{img_path}\n")
        print(f"{filename}: {len(dataset)} entries")

    def run(self):
        self.collect_valid_pairs()
        train_set, val_set, test_set = self.split_dataset()
        self.write_split_files(train_set, val_set, test_set)

# =======================
# 🛠️ Configuration Section
# =======================
if __name__ == "__main__":
    preparer = DatasetPreparer(
        dataset_root="datasets/human_dataset",         # Root folder containing subfolders of data
        output_dir="./output",            # Output folder for train.txt, val.txt, test.txt
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        image_extensions=('.jpg', '.jpeg', '.png'),
        annotation_extension='.txt',
        seed=123
    )
    preparer.run()
