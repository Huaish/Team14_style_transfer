import os
import cv2
from python_color_transfer.color_transfer import ColorTransfer
import subprocess
import shutil

def train_model(image_path, output_dir, prompt, train_script, train_params_template):
    """
    執行單張圖片的模型訓練

    Args:
        image_path (str): 單張圖片的路徑。
        output_dir (str): 訓練結果保存目錄。
        prompt (str): 訓練所用的 prompt。
        train_script (str): 訓練腳本名稱。
        train_params_template (str): 訓練參數模板。
    """
    # 格式化訓練參數
    print("========== training ==========")
    train_params = train_params_template.format(
        instance_data_dir=image_path,
        output_dir=output_dir,
        prompt=prompt
    )
    print("********")
    print(train_params)
    
    try:
        print(f"Starting training for image: {image_path} with prompt: {prompt}")
        subprocess.run(train_params, shell=True, check=True)
        print(f"Training completed and saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed : {e}")

def inference_model(content_lora, style_lora, prompt, output_path):
    """
    執行推理並存放結果

    Args:
        content_lora (str): Content LoRA 模型路徑。
        style_lora (str): Style LoRA 模型路徑。
        prompt (str): 推理所用的 prompt。
        output_path (str): 推理結果保存目錄。
    """
    print("========== inferencing ==========")
    inference_command = (
        f"python inference.py "
        f"--prompt=\"{prompt}\" "
        f"--content_B_LoRA=\"{content_lora}\" "
        f"--style_B_LoRA=\"{style_lora}\" "
        f"--output_path=\"{output_path}\""
    )
    try:
        print(f"Running inference with prompt: {prompt}")
        subprocess.run(inference_command, shell=True, check=True)
        print(f"Inference completed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Inference failed for prompt: {prompt}: {e}")
        
        
def color_transfer_groups(style_root, content_root, output_root, train_script, train_params_template):
    """
    處理圖像並啟動訓練

    Args:
        style_root (str): 風格圖像的根目錄。
        content_root (str): 內容圖像的根目錄。
        output_root (str): 處理結果的根目錄。
        train_script (str): 訓練腳本名稱。
        train_params_template (str): 訓練參數模板。
    """
    os.makedirs(output_root, exist_ok=True)
    ct = ColorTransfer()
    methods = [
        "original",
        "match_histograms",
        "pca_transfer",
        "luv_transfer",
        "lab_transfer",
        "mean_std_transfer",
        "pdf_transfer",
    ]

    # 定義 Style 和 Content 的配對關係
    # group_pairs = [
    #     (1, "style1", "content1"),
    #     (2, "style2", "content2"),
    #     (3, "style3", "content3"),
    #     (4, "style4", "content4"),
    #     (5, "style5", "content1")
    # ]
    group_pairs = [
        (5, "style5", "content5")
    ]

    # 記錄已處理的 Content 圖片，避免重複訓練
    trained_content_images = {}

    for group_id, style_dir, content_dir in group_pairs:
        style_path = os.path.join(style_root, style_dir)
        content_path = os.path.join(content_root, content_dir)
        output_group_dir = os.path.join(output_root, f"group{group_id}")
        os.makedirs(output_group_dir, exist_ok=True)

        if not os.path.isdir(style_path) or not os.path.isdir(content_path):
            print(f"Could not find style or content directory for group {group_id}")
            continue
        
        # raise NotImplementedError

        # 訓練 Content 圖片
        print(f"========== training content for group {group_id}, content path: {content_path} ==========")
        for content_img_name in sorted(os.listdir(content_path)):
            content_img_path = os.path.join(content_path, content_img_name)
            if not os.path.isfile(content_img_path):
                continue

            # 避免重複訓練 Content 圖片
            if content_img_path in trained_content_images:
                print("already exists")
                continue

            content_img = cv2.imread(content_img_path)
            content_base_name = os.path.splitext(content_img_name)[0]

            # 設定 Content 模型輸出目錄
            content_output_dir = os.path.join(
                "lora-dreambooth-model",
                f"Group{group_id}",
                "Content_trained_model",
                f"C{content_base_name}"
            )
            os.makedirs(content_output_dir, exist_ok=True)
            if os.listdir(content_output_dir):
                print(f"Content model for {content_img_name} already exists, skipping...")
                trained_content_images[content_base_name] = content_output_dir
                continue
            
            # 避免重複訓練 Content 圖片
            if content_base_name not in trained_content_images:
                content_prompt = f"A C{content_base_name}"
                train_model(content_img_path, content_output_dir, content_prompt, train_script, train_params_template)
                trained_content_images[content_base_name] = content_output_dir

        # 處理 Style 圖片
        print(f"========== color transfer for group {group_id}, style path: {style_path} ==========")
        for style_img_name in sorted(os.listdir(style_path)):
            style_img_path = os.path.join(style_path, style_img_name)
            if not os.path.isfile(style_img_path):
                continue

            style_img = cv2.imread(style_img_path)

            for content_img_name in sorted(os.listdir(content_path)):
                content_img_path = os.path.join(content_path, content_img_name)
                if not os.path.isfile(content_img_path):
                    continue

                content_img = cv2.imread(content_img_path)
                style_base_name = os.path.splitext(style_img_name)[0]
                content_base_name = os.path.splitext(content_img_name)[0]

                # 創建臨時目錄存儲處理後的圖像
                temp_dir = os.path.join(output_group_dir, f"S{style_base_name}_to_C{content_base_name}")
                os.makedirs(temp_dir, exist_ok=True)

                # 保存原始 style 圖像
                original_style_path = os.path.join(temp_dir, "original.jpg")
                cv2.imwrite(original_style_path, style_img)

                # 生成並保存轉換後的圖像並訓練
                
                for idx, method in enumerate(methods, start=1):
                    print(f"========== processing method {method} ==========")
                    try:
                        if method == "original":
                            transformed_img = style_img
                        else:
                            transformed_img = getattr(ct, method)(img_arr_in=style_img, img_arr_ref=content_img)
                            
                        if transformed_img is None:
                            print(f"Group{group_id} / {style_img_path} in method {method} is not supported")
                            continue

                        # 保存轉換後的圖像
                        output_filename = f"{method}.jpg"
                        output_file_path = os.path.join(temp_dir, output_filename)
                        cv2.imwrite(output_file_path, transformed_img)

                        # 訓練模型
                        print(f"========== training for group {group_id}, style path: {style_path} ==========")
                        # method_prompt = f"A S{style_base_name}_{method}"
                        method_prompt = f"A S{style_base_name}{str(idx).zfill(2)}"
                        train_output_dir = os.path.join(
                            "lora-dreambooth-model",
                            f"Group{group_id}",
                            f"S{style_base_name}_to_C{content_base_name}_trained_model",
                            f"S{style_base_name}_{method}_{str(idx).zfill(2)}"
                        )
                        os.makedirs(train_output_dir, exist_ok=True)
                        
                        if os.listdir(train_output_dir):
                            print(f"Style model for {train_output_dir} already exists, skipping...")
                            continue
                        if style_base_name == "32" or style_base_name == "33":
                            print(f"training for {style_base_name}")
                            train_model(output_file_path, train_output_dir, method_prompt, train_script, train_params_template)
                            print(f"content {content_base_name} inference to S{style_base_name}")

                            inference_output_dir = os.path.join(
                                "/home/siangling/B-LoRA/output",
                                f"group{group_id}",
                                f"S{style_base_name}_to_C{content_base_name}",
                                method,
                            )
                            os.makedirs(inference_output_dir, exist_ok=True)
                            # inference_prompt = f"A C{content_base_name} in S{style_base_name}_{method} style"
                            inference_prompt = f"A C{content_base_name} in S{style_base_name}{str(idx).zfill(2)} style"
                            inference_model(
                                trained_content_images[content_base_name],
                                train_output_dir,
                                inference_prompt,
                                inference_output_dir,
                            )

                    except Exception as e:
                        print(f"Error processing {style_img_path} -> {content_img_path} with {method}: {e}")
                        continue

                # 清理臨時目錄
                # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # 定義路徑
    style_root = "data/Style"
    content_root = "data/Content"
    output_root = "data/Processed"
    
    train_script = "python train_dreambooth_b-lora_sdxl.py"
    
    train_params_template = (
        f"{train_script} "
        "--pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' "
        "--instance_data_dir='{instance_data_dir}' "
        "--instance_prompt='{prompt}' "
        "--output_dir='{output_dir}' "
        "--resolution=1024 "
        "--rank=64 "
        "--train_batch_size=1 "
        "--learning_rate=5e-5 "
        "--lr_scheduler='constant' "
        "--lr_warmup_steps=0 "
        "--max_train_steps=350 "
        "--checkpointing_steps=350 "
        "--seed=0 "
        "--gradient_checkpointing "
        "--use_8bit_adam "
        "--mixed_precision='fp16'"
    )

    # 執行色調轉換處理
    color_transfer_groups(style_root, content_root, output_root, train_script, train_params_template)
