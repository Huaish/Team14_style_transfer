import cv2
import numpy as np
import os

def create_rotated_text_image(text, width, height, font_scale=0.7):
    """
    Create an image with rotated text
    """
    # Create a white image
    text_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate text position
    x = width // 2 + text_height // 2
    y = height // 2 - text_width // 2
    
    # Create a temporary image for text
    temp = np.ones((width, height, 3), dtype=np.uint8) * 255
    cv2.putText(temp, text, (y, x), font, font_scale, (0, 0, 0), thickness)
    
    # Rotate the image
    rotated = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated

def create_hierarchical_comparison(content_image, original_style_image, transfer_style_images, result_images, 
                                 method_names, pair, group, output_path='output/teaser/content_improve/S24_to_C22_comparison.png'):
    """
    Create a hierarchical comparison grid with methods at top and results below
    """
    # Set base paths
    root_group = "output/group" + group
    root_content = "data/Content/content" + group
    root_style = "data/Style/style" + group
    root_process = "data/Processed/group" + group
    
    # Read base images
    content_path = os.path.join(root_content, content_image)
    style_path = os.path.join(root_style, original_style_image)
    
    # Set unified image size and label width
    target_size = (200, 200)
    label_width = 50  # Reduced width for vertical text
    text_height = 40
    
    try:
        # Calculate total width
        total_width = target_size[0] * (len(method_names) + 1)
        
        # Top method name row
        method_header = np.ones((text_height, total_width, 3), dtype=np.uint8) * 255
        label_names = ['input', 'original', 'lab_transfer', 'luv_transfer', 'pca_transfer']
        for idx, method in enumerate(label_names):
            x_pos = (idx) * target_size[0] + 10
            cv2.putText(method_header, method, 
                       (x_pos, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Create style row
        style_row_images = []
        style_img = cv2.resize(cv2.imread(style_path), target_size)
        style_row_images.append(style_img)
        
        for transfer_image in transfer_style_images:
            transfer_path = os.path.join(root_process, pair, transfer_image)
            transfer_img = cv2.resize(cv2.imread(transfer_path), target_size)
            style_row_images.append(transfer_img)
        style_row = np.hstack(style_row_images)

        # Create result row
        result_row_images = []
        content_img = cv2.resize(cv2.imread(content_path), target_size)
        result_row_images.append(content_img)
        
        for method, result_image in zip(method_names, result_images):
            result_path = os.path.join(root_group, pair, method, result_image)
            result_img = cv2.resize(cv2.imread(result_path), target_size)
            result_row_images.append(result_img)
        result_row = np.hstack(result_row_images)

        # Combine right part
        right_part = np.vstack([method_header, style_row, result_row])
        
        # Create rotated labels
        label_height = right_part.shape[0] - text_height  # Subtract header height
        style_label = create_rotated_text_image("Style", label_width, target_size[1])
        content_label = create_rotated_text_image("Content", label_width, target_size[1])
        
        # Create left label area
        left_labels = np.ones((label_height, label_width, 3), dtype=np.uint8) * 255
        
        # Position rotated labels in the middle of their respective rows
        y_offset = text_height
        left_labels[0:target_size[1], :] = style_label
        left_labels[target_size[1]:2*target_size[1], :] = content_label
        
        # Add empty space for the header
        header_space = np.ones((text_height, label_width, 3), dtype=np.uint8) * 255
        left_labels_full = np.vstack([header_space, left_labels])
        
        # Combine final image
        final_image = np.hstack([left_labels_full, right_part])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save result
        cv2.imwrite(output_path, final_image)
        print(f"Saved comparison image to {output_path}")
        
        return final_image

    except Exception as e:
        print(f"Error creating comparison: {str(e)}")
        raise

if __name__ == '__main__':    
    group = "4"
    pair = "S24_to_C22"
    method_names = ['original', 'lab_transfer', 'luv_transfer', 'pca_transfer']
    content_image = '22.jpg'
    original_style_image = '24.jpg'
    transfer_style_images = ['original.jpg', 'lab_transfer.jpg', 'luv_transfer.jpg', 'pca_transfer.jpg']
    result_images = ['A C22 in S2401 style_1.jpg', 'A C22 in S2405 style_0.jpg', 'A C22 in S2404 style_2.jpg', 'A C22 in S2403 style_0.jpg']
    
    create_hierarchical_comparison(content_image, original_style_image, transfer_style_images, 
                                 result_images, method_names, pair, group)