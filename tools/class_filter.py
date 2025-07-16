"""========================================================================================================
#
# ==============================================
# ClassFilter is a Python class for filtering YOLO-style annotation datasets.
# Its main capabilities include:
# 1. Scanning all annotation (.txt) files in a dataset directory and locating corresponding image files
#    (e.g., .jpg, .png, etc.).
# 2. Filtering annotations to retain only the user-specified class IDs, removing others.
# 3. Optionally re-encoding the retained class IDs to new sequential values (e.g., 0,2 → 0,1).
# 4. If an annotation file becomes empty after filtering and `remove_empty=True`,
#    both the annotation and corresponding image file will be deleted.
#
# Use Case:
# This tool is especially useful when training object detection models on a subset of classes,
# ensuring the dataset contains only relevant annotations in proper YOLO format.
# ==============================================
#
# ==============================================
# ClassFilter 是一個用於 YOLO 格式標註資料篩選的 Python 類別。
# 它的主要功能包括：
# 1. 掃描資料夾內所有標註檔（.txt），並找出對應的影像檔（例如 .jpg, .png 等）。
# 2. 根據使用者指定的類別 ID，保留這些類別的標註，移除其他類別。
# 3. 可選擇是否重新編碼保留的類別 ID（例如將 0,2 轉為 0,1），保持類別編號連續。
# 4. 若標註檔在篩選後為空，且設定為 `remove_empty=True`，則會自動刪除對應的標註檔與影像檔。
#
# 使用情境：
# 此工具適用於欲針對部分目標類別進行模型訓練時的資料過濾，
# 特別適合 YOLO 等需精確標註格式的物件偵測應用。
# ==============================================
#
========================================================================================================"""



import os
from typing import List, Dict


class ClassFilter:
    def __init__(self, dataset_dir: str, keep_ids: List[int], reencode: bool = False, remove_empty: bool = False):
        self.dataset_dir = dataset_dir
        self.keep_ids = keep_ids
        self.reencode = reencode
        self.remove_empty = remove_empty
        self.supported_image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    def filter_dataset(self):
        for file_name in os.listdir(self.dataset_dir):
            if file_name.endswith('.txt'):
                txt_path = os.path.join(self.dataset_dir, file_name)
                image_path = self._find_corresponding_image(txt_path)

                if image_path:
                    filtered_lines = self._filter_labels(txt_path)
                    print(f"Processing: {file_name} -> {len(filtered_lines)} valid annotations")

                    if filtered_lines:
                        with open(txt_path, 'w') as f:
                            f.writelines(filtered_lines)
                    elif self.remove_empty:
                        os.remove(txt_path)
                        os.remove(image_path)
                        print(f"Removed: {file_name} and image {os.path.basename(image_path)} (empty annotations)")
                else:
                    print(f"Warning: No image found for {file_name}")


    def _find_corresponding_image(self, txt_path: str) -> str or None:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        for ext in self.supported_image_exts:
            image_path = os.path.join(self.dataset_dir, base_name + ext)
            if os.path.exists(image_path):
                return image_path
        return None

    def _filter_labels(self, txt_path: str) -> List[str]:
        filtered = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id in self.keep_ids:
                if self.reencode:
                    new_class_id = self.keep_ids.index(class_id)
                    parts[0] = str(new_class_id)
                filtered.append(' '.join(parts) + '\n')

        return filtered


# ------------------------------
# ⚙️ User Configuration
# ------------------------------
if __name__ == "__main__":
    dataset_directory = "datasets/human_dataset/coco_dataset"  # e.g., "./dataset"
    ids_to_keep = [0]  # Keep only class 0 and 2
    should_reencode = False  # Reencode 0 -> 0, 2 -> 1
    delete_empty_files = True  # Remove images without relevant objects

    yolo_filter = ClassFilter(
        dataset_dir=dataset_directory,
        keep_ids=ids_to_keep,
        reencode=should_reencode,
        remove_empty=delete_empty_files
    )

    yolo_filter.filter_dataset()
    print("Dataset filtering completed.")
