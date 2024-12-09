import os
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from torch.nn.functional import mse_loss
from scipy.linalg import sqrtm
import pandas as pd
from tqdm import tqdm

def prepare_eval_images(group_id, methods):
    content_dir = f'data/Content/content{group_id}'
    style_dir = f'data/Style/style{group_id}'
    stylized_dir = f'output/before_styleshot/style{group_id}_content{group_id}'

    content_out = f'evaluation/Content/content{group_id}'
    style_out = f'evaluation/Style/style{group_id}'

    os.makedirs(content_out, exist_ok=True)
    os.makedirs(style_out, exist_ok=True)

    content_files = sorted(os.listdir(content_dir))
    style_files = sorted(os.listdir(style_dir))
    for method in methods:
        stylized_out = f'evaluation/Stylized/style{group_id}_content{group_id}/{method}'
        os.makedirs(stylized_out, exist_ok=True)

        for style_file in tqdm(style_files, desc=f'Processing styles for method {method}'):
            for content_file in tqdm(content_files, desc=f'Processing contents for style {style_file}', leave=False):
                content = Image.open(os.path.join(content_dir, content_file)).resize((256, 256))
                style = Image.open(os.path.join(style_dir, style_file)).resize((256, 256))

                content_number = int(content_file.split('.')[0])
                style_number = int(style_file.split('.')[0])
                stylized_path = f"{stylized_dir}/{style_number:02d}_{content_number:02d}_{method}_styleshot.png"
                if not os.path.exists(stylized_path):
                    continue
                
                stylized = Image.open(f"{stylized_dir}/{style_number:02d}_{content_number:02d}_{method}_styleshot.png").resize((256, 256))

                content.save(os.path.join(content_out, f'{style_number:02d}_{content_number:02d}.png'))
                style.save(os.path.join(style_out, f'{style_number:02d}_{content_number:02d}.png'))
                stylized.save(os.path.join(stylized_out, f'{style_number:02d}_{content_number:02d}.png'))
                
class VGG19Extractor(torch.nn.Module):
    def __init__(self):
        super(VGG19Extractor, self).__init__()
        vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features.eval()
        self.layers = {
            '0': 'conv1_1',  # Style layer
            '5': 'conv2_1',  # Style layer
            '10': 'conv3_1', # Style layer
            '19': 'conv4_1', # Style layer
            '21': 'conv4_2', # Content layer
            '28': 'conv5_1'  # Style layer
        }
        self.model = torch.nn.Sequential(*list(vgg)[:29])  # 裁剪模型

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

class VGG16Extractor(torch.nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        vgg = models.vgg16(weights='VGG16_Weights.DEFAULT').features.eval()
        self.layers = {
            '0': 'conv1_1',  # Style layer
            '5': 'conv2_1',  # Style layer
            '10': 'conv3_1', # Style layer
            '17': 'conv4_1', # Style layer
            '19': 'conv4_2', # Content layer
            '24': 'conv5_1'  # Style layer
        }
        self.model = torch.nn.Sequential(*list(vgg)[:29])  # 裁剪模型

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram

def calculate_fid(mean1, cov1, mean2, cov2):
    diff = np.sum((mean1 - mean2) ** 2)
    cov_mean = sqrtm(np.dot(cov1, cov2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real  # 去掉複數部分
    fid = diff + np.trace(cov1 + cov2 - 2 * cov_mean)
    return fid

def preprocess_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        images.append(transform(image).unsqueeze(0))
    return torch.cat(images, dim=0).cuda()

def extract_features_and_calculate_metrics(content_images, style_images, stylized_images, model, batch_size=1):
    num_samples = content_images.size(0)
    content_loss = 0.0
    content_similarity = 0.0
    style_loss_gram = 0.0
    style_loss_similarity = 0.0
    style_features_list = []
    stylized_features_list = []

    for i in range(0, num_samples, batch_size):
        content_batch = content_images[i:i+batch_size]
        style_batch = style_images[i:i+batch_size]
        stylized_batch = stylized_images[i:i+batch_size]
        gray_style_batch = transforms.Grayscale()(style_batch)
        gray_style_batch = gray_style_batch.repeat(1, 3, 1, 1)
        gray_stylized_batch = transforms.Grayscale()(stylized_batch)
        gray_stylized_batch = gray_stylized_batch.repeat(1, 3, 1, 1)

        # 提取特徵
        content_features = model(content_batch)
        # style_features = model(style_batch)
        stylized_features = model(stylized_batch)
        gray_style_features = model(gray_style_batch)
        gray_stylized_features = model(gray_stylized_batch)

        # 累加內容損失
        content_loss += mse_loss(stylized_features['conv4_2'], content_features['conv4_2']).item()
        # content_loss += torch.norm(stylized_features['conv4_2'] - content_features['conv4_2'], p=2).item()
        content_similarity += torch.nn.functional.cosine_similarity(stylized_features['conv4_2'], content_features['conv4_2'], dim=1).mean().item()
        style_loss_similarity += torch.nn.functional.cosine_similarity(gray_stylized_features['conv2_1'], gray_style_features['conv2_1'], dim=1).mean().item()
        
        # 累加風格損失
        style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}
        for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            gram_style = gram_matrix(gray_style_features[layer])
            gram_stylized = gram_matrix(gray_stylized_features[layer])
            b, c, h, w = gray_style_features[layer].size()
            style_loss_gram += style_weights[layer] * torch.sum((gram_style - gram_stylized) ** 2) / (4 * (c ** 2) * (h * w) ** 2)

        # 保存特徵供後續計算 FID
        style_features_list.append(torch.flatten(gray_style_features['conv2_1'], start_dim=2).permute(0, 2, 1))
        stylized_features_list.append(torch.flatten(gray_stylized_features['conv2_1'], start_dim=2).permute(0, 2, 1))

    # 拼接所有批次的特徵
    style_flattened = torch.cat(style_features_list, dim=0).reshape(-1, style_features_list[0].shape[-1]).detach().cpu().numpy()
    stylized_flattened = torch.cat(stylized_features_list, dim=0).reshape(-1, stylized_features_list[0].shape[-1]).detach().cpu().numpy()

    # 計算相似度
    content_similarity /= (num_samples / batch_size)
    style_loss_gram /= (num_samples / batch_size)
    style_loss_similarity /= (num_samples / batch_size)
    
    # 計算 FID
    mean_style = np.mean(style_flattened, axis=0)
    mean_stylized = np.mean(stylized_flattened, axis=0)
    cov_style = np.cov(style_flattened, rowvar=False)
    cov_stylized = np.cov(stylized_flattened, rowvar=False)
    fid = calculate_fid(mean_style, cov_style, mean_stylized, cov_stylized)

    # 返回平均損失與 FID
    content_loss /= (num_samples / batch_size)
    style_loss_gram /= (num_samples / batch_size)
    return content_loss, style_loss_gram, content_similarity, style_loss_similarity, fid

def evaluate_methods(group_id, methods, vgg_extractor):
    content_dir = f"evaluation/Content/content{group_id}"
    style_dir = f"evaluation/Style/style{group_id}"
    results = []
    for method in methods:
        stylized_dir = f"evaluation/Stylized/style{group_id}_content{group_id}/{method}"
        
        content_paths = [os.path.join(content_dir, file) for file in os.listdir(content_dir)]
        style_paths = [os.path.join(style_dir, file) for file in os.listdir(style_dir)]
        stylized_paths = [os.path.join(stylized_dir, file) for file in os.listdir(stylized_dir)]

        # remove the image which does not have stylized image
        content_paths = [path for path in content_paths if os.path.exists(os.path.join(stylized_dir, os.path.basename(path)))]
        style_paths = [path for path in style_paths if os.path.exists(os.path.join(stylized_dir, os.path.basename(path)))]
        stylized_paths = [path for path in stylized_paths if os.path.exists(path)]

        assert len(stylized_paths) != 0
        assert len(content_paths) == len(style_paths) == len(stylized_paths)

        # preprocess images
        content_images = preprocess_images(content_paths)
        style_images = preprocess_images(style_paths)
        stylized_images = preprocess_images(stylized_paths)
        
        # extract features and calculate metrics
        vgg_extractor = vgg_extractor.cuda()
        content_loss, style_loss, content_similarity, style_loss_similarity, fid_score = extract_features_and_calculate_metrics(content_images, style_images, stylized_images, vgg_extractor)

        results.append({
            "Method": method,
            "Content Loss↓": content_loss,
            "Content Similarity↑": content_similarity,
            "Style Loss↓": style_loss.item(),
            "Style Similarity↑": style_loss_similarity,
            "FID↓": fid_score
        })
    
    return results

if __name__ == '__main__':
    # vgg_extractor = VGG19Extractor().eval()
    vgg_extractor = VGG16Extractor().eval()

    # run all groups
    result_all = []

    for id in tqdm(range(1, 6), desc="Processing all groups"):
        group_id = str(id)
        methods = ["original", "lab", "luv", "pca"]
        data_path = os.path.join('evaluation', 'Stylized', f'style{group_id}_content{group_id}')

        if not os.path.exists(data_path):
            prepare_eval_images(group_id, methods)

        result = evaluate_methods(group_id, methods, vgg_extractor)
        result_all.append(result)
        
        # save result
        df = pd.DataFrame(result)
        # save to csv
        df.to_csv(f"evaluation/evaluation{group_id}.csv", index=False)

    # save results
    # Flatten result_all into a single list of dictionaries
    flat_results = [item for sublist in result_all for item in sublist]

    # Convert to a DataFrame for easier processing
    df = pd.DataFrame(flat_results)

    # Columns to compute mean and standard deviation
    metrics = ['Content Loss↓', 'Content Similarity↑', 'Style Loss↓', 'Style Similarity↑', 'FID↓']

    # Initialize a formatted results list
    formatted_results = []

    # Group by 'Method' and calculate mean ± std
    for method, group in df.groupby('Method'):
        formatted_row = {'Method': method}
        for metric in metrics:
            mean = group[metric].mean()
            std = group[metric].std()
            formatted_row[metric] = f"{mean:.3f} ±{std:.3f}"
        formatted_results.append(formatted_row)

    # Create a new DataFrame with formatted results
    formatted_df = pd.DataFrame(formatted_results)


    # Sort the DataFrame by 'Method' in the specified order
    method_order = ['original', 'lab', 'luv', 'pca']
    formatted_df['Method'] = pd.Categorical(formatted_df['Method'], categories=method_order, ordered=True)
    formatted_df = formatted_df.sort_values('Method')
    # Reset the index to reorder it
    formatted_df = formatted_df.reset_index(drop=True)

    # Save to CSV if needed
    formatted_df.to_csv("evaluation/evaluation.csv", index=False)

    # Print each formatted item
    print(formatted_df.to_string(index=False))