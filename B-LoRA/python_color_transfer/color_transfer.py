# -*- coding: utf-8 -*-
""" Implementation of color transfer in python.

Papers: 
    Color Transfer between Images. (2001)
    Automated colour grading using colour distribution transfer. (2007) 
Referenced Implementations:
    https://github.com/chia56028/Color-Transfer-between-Images
    https://github.com/frcs/colour-transfer
"""

import cv2
import numpy as np
from python_color_transfer.utils import Rotations


class ColorTransfer:
    """ Methods for color transfer of images. """

    def __init__(self, eps=1e-6, m=6, c=3):
        """Hyper parameters.

        Attributes:
            c: dim of rotation matrix, 3 for oridnary img.
            m: num of random orthogonal rotation matrices.
            eps: prevents from zero dividing.
        """
        self.eps = eps
        if c == 3:
            self.rotation_matrices = Rotations.optimal_rotations()
        else:
            self.rotation_matrices = Rotations.random_rotations(m, c=c)
        self.RG = Regrain()
        
    def match_histograms(self, img_arr_in, img_arr_ref):
        """Matches the color distribution of the style image to the content image.

        Args:
            style_image: BGR numpy array of style/reference image.
            content_image: BGR numpy array of content/input image.

        Returns:
            Transformed image with matched histograms.
        """
        # Convert images to float for calculations
        img_arr_in = img_arr_in.astype(np.float32)
        img_arr_ref = img_arr_ref.astype(np.float32)

        # Calculate mean and covariance
        def compute_mean_and_covariance(image):
            pixels = image.reshape(-1, 3)
            mean = np.mean(pixels, axis=0)
            covariance = np.cov(pixels, rowvar=False)
            return mean, covariance

        mean_c, cov_c = compute_mean_and_covariance(img_arr_ref)
        mean_s, cov_s = compute_mean_and_covariance(img_arr_in)

        # Compute transformation matrix A using Cholesky decomposition
        L_s = np.linalg.cholesky(cov_s + self.eps * np.eye(3))
        L_c = np.linalg.cholesky(cov_c + self.eps * np.eye(3))
        A = np.dot(np.linalg.inv(L_s), L_c)

        # Apply linear transformation
        transformed_image = np.dot(img_arr_in.reshape(-1, 3) - mean_s, A.T) + mean_c
        transformed_image = transformed_image.reshape(img_arr_in.shape)

        return np.clip(transformed_image, 0, 255).astype(np.uint8)
    
    
    def pca_transfer(self, img_arr_in=None, img_arr_ref=None):
        """Apply PCA-based color transfer from source to target image.

        Args:
            source: BGR numpy array of source image.
            target: BGR numpy array of target image.

        Returns:
            Transferred image with colors matched via PCA.
        """
        # 檢查是否為單色圖像
        if np.std(img_arr_in) < 1e-5 or np.std(img_arr_ref) < 1e-5:
            print("Skipping PCA transfer due to single-color image.")
            return None
        # Convert images to LAB color space
        source_lab = cv2.cvtColor(img_arr_in, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(img_arr_ref, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Compute means of source and target images
        source_mean = np.mean(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))

        # Center the image data
        source_centered = source_lab - source_mean
        target_centered = target_lab - target_mean

        # Compute covariance matrices
        source_cov = np.cov(source_centered.reshape(-1, 3), rowvar=False)
        target_cov = np.cov(target_centered.reshape(-1, 3), rowvar=False)

        # Compute eigenvalues and eigenvectors
        source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
        target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)

        # Construct diagonal square root matrices
        source_diag_sqrt = np.diag(np.sqrt(source_eigvals))
        target_diag_sqrt = np.diag(np.sqrt(target_eigvals))

        # Compute transformation matrix
        transform_matrix = (
            target_eigvecs
            @ target_diag_sqrt
            @ np.linalg.inv(source_diag_sqrt)
            @ np.linalg.inv(source_eigvecs)
        )

        # Apply transformation matrix
        result_centered = np.dot(source_centered.reshape(-1, 3), transform_matrix.T)
        result_lab = result_centered.reshape(source_lab.shape) + target_mean

        # Clip values to valid range
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

        # Convert back to BGR color space
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        return result_bgr
    
    def luv_transfer(self, img_arr_in=None, img_arr_ref=None):
        """Convert img from rgb space to luv space, apply mean std transfer,
        then convert back.
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """
        luv_in = cv2.cvtColor(img_arr_in, cv2.COLOR_BGR2LUV)
        luv_ref = cv2.cvtColor(img_arr_ref, cv2.COLOR_BGR2LUV)
        luv_out = self.mean_std_transfer(img_arr_in=luv_in, img_arr_ref=luv_ref)
        img_arr_out = cv2.cvtColor(luv_out, cv2.COLOR_LUV2BGR)
        return img_arr_out

    # 從 RGB 空間轉換到 LAB 色彩空間
    # LAB 色彩空間由亮度 (L)、綠紅分量 (A)、藍黃分量 (B) 組成
    # 使用 mean_std_transfer，將輸入圖像的平均值與標準差調整為與參考圖像相同。
    def lab_transfer(self, img_arr_in=None, img_arr_ref=None):
        """Convert img from rgb space to lab space, apply mean std transfer,
        then convert back.
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """
        lab_in = cv2.cvtColor(img_arr_in, cv2.COLOR_BGR2LAB)
        lab_ref = cv2.cvtColor(img_arr_ref, cv2.COLOR_BGR2LAB)
        lab_out = self.mean_std_transfer(img_arr_in=lab_in,
                                         img_arr_ref=lab_ref)
        img_arr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        return img_arr_out

    def mean_std_transfer(self, img_arr_in=None, img_arr_ref=None):
        """Adapt img_arr_in's (mean, std) to img_arr_ref's (mean, std).

        img_o = (img_i - mean(img_i)) / std(img_i) * std(img_r) + mean(img_r).
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """
        mean_in = np.mean(img_arr_in, axis=(0, 1), keepdims=True)
        mean_ref = np.mean(img_arr_ref, axis=(0, 1), keepdims=True)
        std_in = np.std(img_arr_in, axis=(0, 1), keepdims=True)
        std_ref = np.std(img_arr_ref, axis=(0, 1), keepdims=True)
        img_arr_out = (img_arr_in - mean_in) / std_in * std_ref + mean_ref
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 255] = 255
        return img_arr_out.astype("uint8")

    def pdf_transfer(self, img_arr_in=None, img_arr_ref=None, regrain=False):
        """Apply probability density function transfer.

        img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
        where f_{img}(r, g, b) is the probability density function of img's rgb values.
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """

        # reshape (h, w, c) to (c, h*w)
        [h, w, c] = img_arr_in.shape
        reshape_arr_in = img_arr_in.reshape(-1, c).transpose() / 255.0
        reshape_arr_ref = img_arr_ref.reshape(-1, c).transpose() / 255.0
        # pdf transfer
        reshape_arr_out = self.pdf_transfer_nd(arr_in=reshape_arr_in,
                                               arr_ref=reshape_arr_ref)
        # reshape (c, h*w) to (h, w, c)
        reshape_arr_out[reshape_arr_out < 0] = 0
        reshape_arr_out[reshape_arr_out > 1] = 1
        reshape_arr_out = (255.0 * reshape_arr_out).astype("uint8")
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
        if regrain:
            img_arr_out = self.RG.regrain(img_arr_in=img_arr_in,
                                          img_arr_col=img_arr_out)
        return img_arr_out

    # 將輸入數據的多維概率密度分佈調整為與參考數據的分佈一致。
    def pdf_transfer_nd(self, arr_in=None, arr_ref=None, step_size=1):
        """Apply n-dim probability density function transfer.

        Args:
            arr_in: shape=(n, x).
            arr_ref: shape=(n, x).
            step_size: arr = arr + step_size * delta_arr.
        Returns:
            arr_out: shape=(n, x).
        """
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, arr_out)
            rot_arr_ref = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i],
                                                       rot_arr_ref[i])
            # func = lambda x, n : self._pdf_transfer_1d(x[:n], x[n:])
            # rot_arr = np.concatenate((rot_arr_in, rot_arr_ref), axis=1)
            # rot_arr_out = np.apply_along_axis(func, 1, rot_arr, rot_arr_in.shape[1])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(
                rotation_matrix.transpose(), rot_delta_arr
            )  # np.linalg.solve(rotation_matrix, rot_delta_arr)
            arr_out = step_size * delta_arr + arr_out
        return arr_out

    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None, n=300):
        """Apply 1-dim probability density function transfer.

        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
            n: discretization num of distribution of image's pixels.
        Returns:
            arr_out: transfered input array.
        """

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # transfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in <= d_ref[0]] = min_v
        t_d_in[d_in >= d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out


class Regrain:

    def __init__(self, smoothness=1):
        """To understand the meaning of these params, refer to paper07."""
        self.nbits = [4, 16, 32, 64, 64, 64]
        self.smoothness = smoothness
        self.level = 0

    def regrain(self, img_arr_in=None, img_arr_col=None):
        """keep gradient of img_arr_in and color of img_arr_col. """

        img_arr_in = img_arr_in / 255.0
        img_arr_col = img_arr_col / 255.0
        img_arr_out = np.array(img_arr_in)
        img_arr_out = self.regrain_rec(img_arr_out, img_arr_in, img_arr_col,
                                       self.nbits, self.level)
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 1] = 1
        img_arr_out = (255.0 * img_arr_out).astype("uint8")
        return img_arr_out

    def regrain_rec(self, img_arr_out, img_arr_in, img_arr_col, nbits, level):
        """direct translation of matlab code. """

        [h, w, _] = img_arr_in.shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            resize_arr_in = cv2.resize(img_arr_in, (w2, h2),
                                       interpolation=cv2.INTER_LINEAR)
            resize_arr_col = cv2.resize(img_arr_col, (w2, h2),
                                        interpolation=cv2.INTER_LINEAR)
            resize_arr_out = cv2.resize(img_arr_out, (w2, h2),
                                        interpolation=cv2.INTER_LINEAR)
            resize_arr_out = self.regrain_rec(resize_arr_out, resize_arr_in,
                                              resize_arr_col, nbits[1:],
                                              level + 1)
            img_arr_out = cv2.resize(resize_arr_out, (w, h),
                                     interpolation=cv2.INTER_LINEAR)
        img_arr_out = self.solve(img_arr_out, img_arr_in, img_arr_col,
                                 nbits[0], level)
        return img_arr_out

    def solve(self,
              img_arr_out,
              img_arr_in,
              img_arr_col,
              nbit,
              level,
              eps=1e-6):
        """direct translation of matlab code. """

        [width, height, c] = img_arr_in.shape
        first_pad_0 = lambda arr: np.concatenate(
            (arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr: np.concatenate(
            (arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr: np.concatenate(
            (arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr: np.concatenate(
            (arr[:, 1:], arr[:, -1:]), axis=1)

        delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
        delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
        delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

        psi = 256 * delta / 5
        psi[psi > 1] = 1
        phi = 30 * 2**(-level) / (1 + 10 * delta / self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1 / 5.0
        for i in range(nbit):
            den = psi + phi1 + phi2 + phi3 + phi4
            num = (
                np.tile(psi, [1, 1, c]) * img_arr_col +
                np.tile(phi1, [1, 1, c]) *
                (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in)
                + np.tile(phi2, [1, 1, c]) *
                (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in)
                + np.tile(phi3, [1, 1, c]) *
                (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) +
                 img_arr_in) + np.tile(phi4, [1, 1, c]) *
                (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) +
                 img_arr_in))
            img_arr_out = (num / np.tile(den + eps, [1, 1, c]) * (1 - rho) +
                           rho * img_arr_out)
        return img_arr_out

if __name__ == "__main__":
    ct = ColorTransfer()
    img_ref = cv2.imread("./data/Content/content1/01.jpg")
    img_in = cv2.imread("./data/Style/style1/01.jpg")
    img_out = ct.pdf_transfer(img_arr_in=img_in, img_arr_ref=img_ref, regrain=False)
    cv2.imwrite("output.jpg", img_out)