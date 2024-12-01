# Blue Bottle - Style Transfer

> Image processing final project  
> TEAM 14

## Installation

### Option 1: Using `venv`

1. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   ```
2. Activate the environment:
   - **Mac/Linux**:
     ```bash
     source myenv/bin/activate
     ```
   - **Windows**:
     ```cmd
     myenv\Scripts\activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using `conda`

1. Create a Conda environment:
   ```bash
   conda create -n bluebottle python=3.9 -y
   ```
2. Activate the environment:
   ```bash
   conda activate bluebottle
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets

The datasets required for the project are stored in the data directory.
The directory contains two main categories: `Content` and `Style`, each containing subdirectories with image files.

## Color Transfer Methods

The `color_transfer.py` script includes the following methods:

1. **match_histograms**: Matches histograms of source and target images.
2. **lab_transfer**: Transfers color in LAB color space.
3. **luv_transfer**: Transfers color in LUV color space.
4. **mean_std_transfer**: Matches mean and standard deviation of colors.
5. **pca_transfer**: Applies Principal Component Analysis (PCA) for color alignment.
6. **pdf_transfer**: Performs probability density function (PDF)-based color transfer.

### Usage

To apply color transfer, use the following code snippet:

```python
from color_transfer import match_histograms, lab_transfer
import cv2

ct = ColorTransfer()
img_in = cv2.imread("data/Content/content1/01.jpg")
img_ref = cv2.imread("data/Style/style1/01.jpg")
img_out = ct.pdf_transfer(img_arr_in=img_in, img_arr_ref=img_ref, regrain=False)
cv2.imwrite("output.jpg", img_out)
```

There is a example in `color_transfer.py` that demonstrates how to use the color transfer methods. To run the example, execute the following command:

```bash
python color_transfer.py
```

Select a method based on your needs for color transformation.

## Demo Website

Launch an interactive web interface to experiment with color transfer:

```python
python demo.py
```

Once started, the web interface can be accessed at [http://127.0.0.1:7861](http://127.0.0.1:7861)
