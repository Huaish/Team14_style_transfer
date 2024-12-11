import os
import cv2
import numpy as np

class ImagePairProcessor:
    def __init__(self, group_dir, style_dir, style_processed_dir, content_dir, output_dir):
        """
        初始化配對處理器

        Args:
            group_dir (str): Group 資料夾路徑 (e.g., "output/group1")
            style_dir (str): 原始 Style 圖片的根目錄 (e.g., "data/Style")
            style_processed_dir (str): 已轉換 Style 圖片的根目錄 (e.g., "data/Processed/group1")
            content_dir (str): Content 圖片的根目錄 (e.g., "data/Content")
            output_dir (str): 最終生成比較圖的存儲路徑 (e.g., "output/group1/comparisons")
        """
        self.group_dir = group_dir
        self.style_dir = style_dir
        self.style_processed_dir = style_processed_dir
        self.content_dir = content_dir
        self.output_dir = output_dir
        self.group_id = int(style_processed_dir.split("/")[-1][-1])  # 提取最後一個數字

    def add_labels(self, image, labels, position=(10, 50), label_width=200, font_scale=1, font_thickness=2):
        """
        在圖片上添加文字標籤，並調整位置對齊

        Args:
            image (np.ndarray): 圖片
            labels (list): 標籤文字列表
            position (tuple): 第一個標籤的起始位置
            label_width (int): 每個標籤的寬度，控制水平間距
            font_scale (int): 字體大小
            font_thickness (int): 字體粗細
        """
        x, y = position
        for label in labels:
            cv2.putText(
                image, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness
            )
            x += label_width  # 水平移動到下一個標籤的位置
    
    def process_pair(self, pair_name):
        """
        處理單個配對並生成對比圖

        Args:
            pair_name (str): 配對名稱 (e.g., "S01_to_C01")
        """
        style_folder = f"style{self.group_id}"  # 動態匹配 Style 資料夾
        # content_folder = f"content{self.group_id}"  # 動態匹配 Content 資料夾
        content_folder = f"content1"  # 動態匹配 Content 資料夾
        pair_processed_dir = os.path.join(self.style_processed_dir, pair_name)
        pair_output_dir = os.path.join(self.group_dir, pair_name)
        style_id, content_id = pair_name.split("_to_")

        # 確定原始 Style 和 Content 圖片路徑
        original_style_path = os.path.join(self.style_dir, style_folder, f"{style_id[-2:]}.jpg")
        content_path = os.path.join(self.content_dir, content_folder, f"{content_id[-2:]}.jpg")

        # 讀取原始 Style 和 Content 圖片
        original_style_img = cv2.imread(original_style_path)
        content_img = cv2.imread(content_path)

        if original_style_img is None or content_img is None:
            print(f"Missing original style or content image for {pair_name}")
            return

        # 初始化結果行列表
        rows = []
        total_width = 1600  # 每行的總寬度（包含左側標籤區域）
        # 添加標題行
        header_labels = ["Method", "Content", "Original Style", "Transfer Style", "Result 1", "Result 2", "Result 3", "Result 4"]
        header_img = np.ones((50, 1600, 3), dtype=np.uint8) * 255  # 預留標題區域，背景改為白色
        self.add_labels(header_img, header_labels, position=(10, 30), label_width=200, font_scale=0.8, font_thickness=2)
        rows.append(header_img)
        
        # 遍歷每種轉換方法的資料夾
        for method in sorted(os.listdir(pair_processed_dir)):
            processed_style_path = os.path.join(pair_processed_dir, method)
            method_name = os.path.splitext(method)[0]
            method_output_dir = os.path.join(pair_output_dir, method_name)

            if not os.path.exists(processed_style_path) or not os.path.exists(method_output_dir):
                print(f"Missing data for method {method} in {pair_name}")
                continue

            # 讀取轉換後的 Style 圖片
            processed_style_img = cv2.imread(processed_style_path)

            # 讀取每種方法的結果圖
            result_images = []
            for result_file in sorted(os.listdir(method_output_dir)):
                result_path = os.path.join(method_output_dir, result_file)
                result_img = cv2.imread(result_path)
                if result_img is not None:
                    result_images.append(result_img)
                    
            # 確保每行的圖片數量和尺寸一致
            if len(result_images) > 0:
                method_row = [content_img, original_style_img, processed_style_img] + result_images
            else:
                # 如果該方法沒有結果圖片，跳過該方法
                print(f"No result images for method {method} in {pair_name}")
                continue

            # 確保所有圖片尺寸一致（調整為 200x200）
            method_row = [cv2.resize(img, (200, 200)) for img in method_row]
            
            # 添加方法名稱作為標籤圖片
            method_label = np.ones((200, 200, 3), dtype=np.uint8) * 255  # 白色背景
            self.add_labels(method_label, [method_name], position=(10, 100), font_scale=0.6, font_thickness=2)
            method_row = [method_label] + method_row  # 在最左邊加上標籤
            
            rows.append(np.hstack(method_row))
            
        # 確保每行寬度一致
        for i in range(len(rows)):
            if rows[i].shape[1] < total_width:
                diff = total_width - rows[i].shape[1]
                padding = np.ones((rows[i].shape[0], diff, 3), dtype=np.uint8) * 255  # 用白色填充
                rows[i] = np.hstack([rows[i], padding])

        final_image = np.vstack(rows)

        # 保存比較圖
        comparison_output_path = os.path.join(self.output_dir, f"{pair_name}_comparison.jpg")
        os.makedirs(os.path.dirname(comparison_output_path), exist_ok=True)
        cv2.imwrite(comparison_output_path, final_image)
        print(f"Comparison image saved to {comparison_output_path}")

    def process_all_pairs(self):
        """
        處理 Group 資料夾中的所有配對
        """
        # self.process_pair("S01_to_C01")
        for pair_name in sorted(os.listdir(self.style_processed_dir)):
            pair_processed_dir = os.path.join(self.style_processed_dir, pair_name)
            if os.path.isdir(pair_processed_dir):
                print(f"Processing pair: {pair_name}")
                self.process_pair(pair_name)


if __name__ == "__main__":
    # 定義路徑
    group_dir = "output/group6"
    style_dir = "data/Style"
    style_processed_dir = "data/Processed/group5"
    content_dir = "data/Content"
    output_dir = "output/group6/comparisons"

    processor = ImagePairProcessor(group_dir, style_dir, style_processed_dir, content_dir, output_dir)
    processor.process_all_pairs()
