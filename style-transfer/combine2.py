from PIL import Image

def combine_images_with_matching_size(image_paths, output_path):
    def resize_image(img, target_size):
        """調整圖片至指定大小"""
        return img.resize(target_size, Image.Resampling.LANCZOS)  # 使用新的縮放方法

    def combine_row(images):
        """水平拼接圖片"""
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)
        row_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width
        return row_image

    # 將圖片分成兩組
    group1 = image_paths[0]
    group2 = image_paths[1]

    # 打開兩組圖片
    images1 = [Image.open(img) for img in group1]
    images2 = [Image.open(img) for img in group2]

    # 獲取 row2 中所有圖片的大小
    row2_widths, row2_heights = zip(*(img.size for img in images2))
    target_widths = list(row2_widths)
    target_heights = list(row2_heights)

    # 將 row1 的圖片調整為與 row2 對應圖片相同大小
    resized_images1 = [
        resize_image(img, (target_widths[i], target_heights[i]))
        for i, img in enumerate(images1)
    ]

    # 拼接每一行
    row1 = combine_row(resized_images1)
    row2 = combine_row(images2)

    # 垂直拼接兩行
    max_width = max(row1.width, row2.width)
    total_height = row1.height + row2.height
    combined_image = Image.new('RGB', (max_width, total_height))
    combined_image.paste(row1, (0, 0))
    combined_image.paste(row2, (0, row1.height))

    # 保存結果
    combined_image.save(output_path)
    print(f"已保存組合圖片到 {output_path}")


content_dir = 'content4'
content_num = '23'
style_dir = 'style4'
style_num = '23'

# 使用
image_files = [
    [f'data/Style/{style_dir}/{style_num}.jpg', 
     f'transfered_style/{content_dir}/{content_num}/style_{style_num}/origin.jpg',
     f'transfered_style/{content_dir}/{content_num}/style_{style_num}/lab.jpg',
     f'transfered_style/{content_dir}/{content_num}/style_{style_num}/luv.jpg'], 
    [f'data/Content/{content_dir}/{content_num}.jpg',
     f'result/{content_dir}/{content_num}/style_{style_num}/preserve_color.jpg',
     f'result/{content_dir}/{content_num}/style_{style_num}/lab.jpg',
     f'result/{content_dir}/{content_num}/style_{style_num}/luv.jpg']
]
combine_images_with_matching_size(image_files, f'combine_img/{content_num}_{style_num}.jpg')
