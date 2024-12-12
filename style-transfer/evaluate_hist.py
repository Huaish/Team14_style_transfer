import os

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim


def histogram_similarity(content_image, generated_image):
    content_image = cv2.imread(content_image_path)
    generated_image = cv2.imread(generated_image_path)
    similarity = 0
    for channel in range(3):  # 遍歷 RGB 三個通道
        hist_content = cv2.calcHist([content_image], [channel], None, [256], [0, 256])
        hist_generated = cv2.calcHist([generated_image], [channel], None, [256], [0, 256])
        hist_content = cv2.normalize(hist_content, hist_content).flatten()
        hist_generated = cv2.normalize(hist_generated, hist_generated).flatten()
        similarity += cv2.compareHist(hist_content, hist_generated, cv2.HISTCMP_CORREL)
    return similarity / 3  # 返回三個通道的平均相似度


if __name__ == '__main__':
    object = 'content'
    
    content_dirs = ["content1", "content2", "content3", "content4", "content5"]

    content_nums_list = [
        ["01", "02", "03", "04", "05", "06"],
        ["07", "08", "09", "10", "11", "12"],
        ["13", "14", "15", "16", "17", "18", "19"],
        ["20", "21", "22", "23", "24", "25", "26"],
        ["27", "28", "29", "30", "31", "32", "33", "34"]            
                         ]

    style_dirs_list = [
        ["style_01", "style_02", "style_03", "style_04", "style_05", "style_06"],
        ["style_07", "style_08", "style_09", "style_10", "style_11", "style_12"],
        ["style_13", "style_14", "style_15", "style_16", "style_17", "style_18"],
        ["style_19", "style_20", "style_21", "style_22", "style_23", "style_24", "style_25"],
        ["style_26", "style_27", "style_28", "style_29", "style_30", "style_31", "style_32", "style_33"]
    ]

    preprocess_types = ["preserve_color", "lab", "luv"]
    # preprocess_types = ["origin", "preserve_color", "lab", "luv", "match_hist", "mean_std", "pca", "pdf"]
    path = os.path.abspath(os.getcwd())
    
    result = []
    for preprocess_type in preprocess_types:
        data = []
        for content_dir in content_dirs:
            content_nums = content_nums_list[int(content_dir.replace("content", "")) - 1]
            style_dirs = style_dirs_list[int(content_dir.replace("content", "")) - 1]
            for content_num in content_nums:
                for style_dir in style_dirs:
                    content_image_path = os.path.join(path, f'data/Content/{content_dir}/{content_num}.jpg')
                    generated_image_path = f"result/{content_dir}/{content_num}/{style_dir}/{preprocess_type}.jpg"
                    data.append(histogram_similarity(content_image_path, generated_image_path))
        data = np.array(data)
        mean = np.mean(data)
        stdev = np.std(data)
        result.append(f'{mean:.3f} ± {stdev:.3f}')
    df = pd.DataFrame({'HIST': result}, index=[preprocess_types])
    df.to_excel("results_HIST.xlsx", index=True)
        