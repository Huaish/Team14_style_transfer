from color_transfer import ColorTransfer
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches
import random

def run_styleshot(style_path, content_path, output_path, seed=0):
    command = f"conda run -n styleshot python inference.py --style ../{style_path} --content ../{content_path} --output ../{output_path} --seed {seed}"
    
    try:
        subprocess.run(command, shell=True, cwd="Styleshot", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error running StyleShot: {e}")
        
    # Load the generated output image
    if os.path.exists(output_path):
        output_image = cv2.imread(output_path)
        return output_image
    else:
        style_image = cv2.imread(style_path)
        return np.zeros_like(style_image)
    
def plot_results(results, cols=8, save_path=None, xlabels=[], ylabels=[], BGR=True):
    rows = len(results) // cols + (1 if len(results) % cols != 0 else 0)
    fig = plt.figure(figsize=(5*cols, 5*rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.2))
    
    for i, (ax, result) in enumerate(zip(grid, results)):
        row_idx = i // cols
        col_idx = i % cols
        
        if BGR:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        ax.imshow(result)
        ax.axis('off')
        
        rect = patches.Rectangle(
            (0, 0), 1, 1,
            linewidth=1, edgecolor="black", facecolor="none", transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        if row_idx == 0 and col_idx < len(xlabels):
            ax.set_title(xlabels[col_idx], fontsize=12, color="black")
        
        if col_idx == 0 and row_idx < len(ylabels):
            ax.annotate(ylabels[row_idx], xy=(-0.1, 0.5), xycoords="axes fraction", fontsize=16, color="black", ha="right", va="center", rotation=90)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    plt.clf()

def color_transfer_before_styleshot(style_folder, content_folder, output_root, seed=0, reverse=False):
    
    style_groups = sorted(os.listdir(style_folder), reverse=reverse)
    content_groups = sorted(os.listdir(content_folder), reverse=reverse)
    
    results_folder = os.path.join(output_root, "results2")
    os.makedirs(results_folder, exist_ok=True)
    
    ct = ColorTransfer()
    ct_methods = [
        # original
        lambda style_path, content_path: style_path,
        ct.match_histograms, ct.pca_transfer, ct.lab_transfer, ct.luv_transfer, ct.mean_std_transfer, ct.pdf_transfer]
    # ct_names = ['original', 'mh', 'pca', 'lab', 'luv', 'mean_std', 'pdf']
    ct_methods = [
        lambda style_path, content_path: style_path,
        ct.lab_transfer, ct.luv_transfer, ct.pca_transfer]
    ct_names = ['original', 'lab', 'luv', 'pca']
    
    for i, style_group in enumerate(style_groups):
        for j, content_group in enumerate(content_groups):
            print(f"Processing {style_group} and {content_group}")
            if i != j:
                continue
            output_folder = os.path.join(output_root, f"{style_group}_{content_group}")
            os.makedirs(output_folder, exist_ok=True)

            styles = sorted(os.listdir(os.path.join(style_folder, style_group)))
            contents = sorted(os.listdir(os.path.join(content_folder, content_group)))
            
            for style_file in styles:
                results = []
                with tqdm(contents, desc="Processing content images", total=len(contents)*len(ct_methods)) as pbar:
                    for content_file in pbar:
                        style_image_name = style_file.split(".")[0]
                        content_image_name = content_file.split(".")[0]
                        
                        style_path = os.path.join(style_folder, style_group, style_file)
                        content_path = os.path.join(content_folder, content_group, content_file)
                        style_image = cv2.imread(style_path)
                        content_image = cv2.imread(content_path)

                        # resize style image to match content image
                        style_image = cv2.resize(style_image, (content_image.shape[1], content_image.shape[0]))
                        style_content = np.vstack([style_image, content_image])
                        results.append(style_content)

                        for ct_name, ct_method in zip(ct_names, ct_methods):
                            pbar.set_postfix({"style": style_image_name, "content": content_image_name, "ct": ct_name})
                            try:
                                transfered = ct_method(style_image, content_image)
                                output_path = os.path.join(output_folder, f"{style_image_name}_{content_image_name}_{ct_name}.png")
                                cv2.imwrite(output_path, transfered)
                                
                                # Run StyleShot
                                style_path = output_path
                                output_path = os.path.join(output_folder, f"{style_image_name}_{content_image_name}_{ct_name}_styleshot.png")
                                if os.path.exists(output_path):
                                    styleshot_result = cv2.imread(output_path)
                                    # print(f"Loaded existing styleshot result for {style_image_name}_{content_image_name}_{ct_name}")
                                else:
                                    styleshot_result = run_styleshot(style_path, content_path, output_path, seed=seed)
                                
                                # vertical concat of color transfer and styleshot results
                                # resize styleshot result to match content image
                                styleshot_result = cv2.resize(styleshot_result, (content_image.shape[1], content_image.shape[0]))
                                result = np.vstack([transfered, styleshot_result])
                            
                            except Exception as e:
                                print(f"Error processing {style_image_name}_{content_image_name}_{ct_name}: {e}")
                                result = np.vstack([np.zeros_like(content_image), np.zeros_like(content_image)])
                            
                            results.append(result)
                            pbar.update(1)
                    
                    plot_results(results, cols=len(ct_names)+1, 
                                 save_path=os.path.join(results_folder, f"style{style_image_name}_{content_group}.png"),
                                 xlabels=["Input"] + ct_names,
                                 ylabels=[f"Content {contents[i].split('.')[0]}    |    Style {style_image_name}" for i in range(len(contents))])

def color_transfer_after_styleshot(style_folder, content_folder, output_root, seed=0, reverse=False):
    
    style_groups = sorted(os.listdir(style_folder), reverse=reverse)
    content_groups = sorted(os.listdir(content_folder), reverse=reverse)
    
    ct = ColorTransfer()
    ct_methods = [
        # original
        lambda style_path, content_path: style_path,
        ct.match_histograms, ct.pca_transfer, ct.lab_transfer, ct.luv_transfer, ct.mean_std_transfer, ct.pdf_transfer]
    ct_names = ['original', 'mh', 'pca', 'lab', 'luv', 'mean_std', 'pdf']
    
    results_folder = os.path.join(output_root, "results2")
    os.makedirs(results_folder, exist_ok=True)

    for i, style_group in enumerate(style_groups):
        for j, content_group in enumerate(content_groups):
            if i != j:
                continue
            output_folder = os.path.join(output_root, f"{style_group}_{content_group}")
            os.makedirs(output_folder, exist_ok=True)

            styles = sorted(os.listdir(os.path.join(style_folder, style_group)))
            contents = sorted(os.listdir(os.path.join(content_folder, content_group)))

            for style_file in styles:
                results = []
                with tqdm(contents, desc="Processing content images", total=len(contents)*len(ct_methods)) as pbar:
                    for content_file in pbar:
                        style_image_name = style_file.split(".")[0]
                        content_image_name = content_file.split(".")[0]
                        
                        style_path = os.path.join(style_folder, style_group, style_file)
                        content_path = os.path.join(content_folder, content_group, content_file)
                        style_image = cv2.imread(style_path)
                        content_image = cv2.imread(content_path)
                        
                        # resize style image to match content image
                        style_image = cv2.resize(style_image, (content_image.shape[1], content_image.shape[0]))
                        
                        results.append(style_image)
                        results.append(content_image)

                        # Run StyleShot
                        output_path = os.path.join(output_folder, f"{style_image_name}_{content_image_name}_styleshot.png")
                        
                        if os.path.exists(output_path):
                            styleshot_result = cv2.imread(output_path)
                            # print(f"Loaded existing styleshot result for {style_image_name}_{content_image_name}")
                        else:
                            styleshot_result = run_styleshot(style_path, content_path, output_path, seed=seed)

                        for ct_name, ct_method in zip(ct_names, ct_methods):
                            pbar.set_postfix({"style": style_image_name, "content": content_image_name, "ct": ct_name})
                            try:
                                # color transfer
                                output_path = os.path.join(output_folder, f"{style_image_name}_{content_image_name}_{ct_name}_styleshot.png")
                                transfered = ct_method(styleshot_result, content_image)
                                output_path = os.path.join(output_folder, f"{style_image_name}_{content_image_name}_{ct_name}.png")
                                cv2.imwrite(output_path, transfered)
                            
                            except Exception as e:
                                print(f"Error processing {style_image_name}_{content_image_name}_{ct_name}: {e}")
                                transfered = np.zeros_like(content_image)
                            
                            results.append(transfered)
                            pbar.update(1)
                    
                    
                    plot_results(results, cols=len(ct_names)+2, 
                                 save_path=os.path.join(results_folder, f"style{style_image_name}_{content_group}.png"),
                                 xlabels=["Style", "Content"] + ct_names,
                                 ylabels=[f"Content {contents[i].split('.')[0]}    |    Style {style_image_name}" for i in range(len(contents))])

if __name__ == "__main__":
    # random seed
    seed = random.randint(0, 1000)
    
    style_folder = "data/Style"
    content_folder = "data/Content"
    output_root = "output/styleshot"
    
    os.makedirs(output_root, exist_ok=True)

    color_transfer_before_styleshot(style_folder, content_folder, output_root, seed=seed, reverse=False)
    