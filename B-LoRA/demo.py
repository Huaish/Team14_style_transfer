# -*- coding: utf-8 -*-

import os
import time

import cv2
import numpy as np

from python_color_transfer.color_transfer import ColorTransfer


def demo():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    img_folder = os.path.join(cur_dir, "my_data")
    style_folder = os.path.join(cur_dir, "style")
    output_folder = os.path.join(cur_dir, "outputs")
    
    img_names = [
        "style02.jpg",
    ]
    ref_names = [
        "img2.JPG",
    ]
    out_names = [
        "style2_img2.png",
    ]
    img_paths = [os.path.join(style_folder, x) for x in img_names]
    ref_paths = [os.path.join(img_folder, x) for x in ref_names]
    out_paths = [os.path.join(output_folder, x) for x in out_names]

    # cls init
    PT = ColorTransfer()

    for img_path, ref_path, out_path in zip(img_paths, ref_paths, out_paths):
        # read input img
        img_arr_in = cv2.imread(img_path)
        [h, w, c] = img_arr_in.shape
        print(f"{img_path}: {h}x{w}x{c}")
        # read reference img
        img_arr_ref = cv2.imread(ref_path)
        [h, w, c] = img_arr_ref.shape
        print(f"{ref_path}: {h}x{w}x{c}")
        # pdf transfer
        t0 = time.time()
        img_arr_reg = PT.pdf_transfer(img_arr_in=img_arr_in,
                                      img_arr_ref=img_arr_ref,
                                      regrain=True)
        print(f"Pdf transfer time: {time.time() - t0:.2f}s")
        # mean transfer
        t0 = time.time()
        img_arr_mt = PT.mean_std_transfer(img_arr_in=img_arr_in,
                                          img_arr_ref=img_arr_ref)
        print(f"Mean std transfer time: {time.time() - t0:.2f}s")
        # lab transfer
        t0 = time.time()
        img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in,
                                     img_arr_ref=img_arr_ref)
        print(f"Lab mean std transfer time: {time.time() - t0:.2f}s")
        
        # Define the target height and width
        target_height = min(img_arr_in.shape[0], img_arr_ref.shape[0], img_arr_mt.shape[0], img_arr_lt.shape[0], img_arr_reg.shape[0])
        target_width = min(img_arr_in.shape[1], img_arr_ref.shape[1], img_arr_mt.shape[1], img_arr_lt.shape[1], img_arr_reg.shape[1])
        
        # Resize all images to the target dimensions
        img_arr_in = cv2.resize(img_arr_in, (target_width, target_height))
        img_arr_ref = cv2.resize(img_arr_ref, (target_width, target_height))
        img_arr_mt = cv2.resize(img_arr_mt, (target_width, target_height))
        img_arr_lt = cv2.resize(img_arr_lt, (target_width, target_height))
        img_arr_reg = cv2.resize(img_arr_reg, (target_width, target_height))
        
        # display
        img_arr_out = np.concatenate(
            (img_arr_in, img_arr_ref, img_arr_mt, img_arr_lt, img_arr_reg),
            axis=1)
        cv2.imwrite(out_path, img_arr_out)
        cv2.imwrite(os.path.join(output_folder, "pdf.png"), img_arr_reg)
        print(f"Saved to {out_path}\n")


if __name__ == "__main__":
    demo()
