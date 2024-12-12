import cv2
import numpy as np
import os
import skimage.io
import tensorflow as tf

import vgg19

# 在執行前需要在檔案中加入 vgg19.npy，下載位址: https://github.com/machrisaa/tensorflow-vgg?tab=readme-ov-file

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

ALPHA = 5.0
BETA = 50.0
LR = 1.0

def load_image(path):
    img = skimage.io.imread(path)
    yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
    img = img - vgg19.VGG_MEAN
    img = img[:, :, (2, 1, 0)]  # rgb to bgr
    return img[np.newaxis, :, :, :], yuv

def save_image(img, path, content_yuv=None):
    img = np.squeeze(img)
    img = img[:, :, (2, 1, 0)]  # bgr to rgb
    img = img + vgg19.VGG_MEAN
    if content_yuv is not None:
        yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
        yuv[:, :, 1:3] = content_yuv[:, :, 1:3]
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    img = np.clip(img, 0, 255).astype(np.uint8)
    skimage.io.imsave(path, img)

def feature_to_gram(f):
    shape = f.get_shape().as_list()
    n_channels = shape[3]
    size = np.prod(shape)
    f = tf.reshape(f, [-1, n_channels])
    return tf.matmul(tf.transpose(f), f) / size

def get_style_rep(vgg):
    return list(map(feature_to_gram, [getattr(vgg, l) for l in STYLE_LAYERS]))

def compute_style_loss(style_rep, image_vgg):
    style_rep_image = get_style_rep(image_vgg)
    style_losses = [tf.nn.l2_loss(a - b) / tf.size(a, out_type=tf.float32) 
                    for a, b in zip(style_rep, style_rep_image)]
    return tf.reduce_sum(style_losses)

def main(content_path, style_path, output_path, iterations, vgg_path, preserve_color):
    content_img, content_yuv = load_image(content_path)
    style_img, _ = load_image(style_path)

    tf.compat.v1.disable_eager_execution()

    with tf.compat.v1.Session() as sess:
        content_vgg = vgg19.Vgg19(vgg_path)
        content = tf.compat.v1.placeholder(tf.float32, content_img.shape)
        content_vgg.build(content)

        style_vgg = vgg19.Vgg19(vgg_path)
        style = tf.compat.v1.placeholder(tf.float32, style_img.shape)
        style_vgg.build(style)

        sess.run(tf.compat.v1.global_variables_initializer())
        content_rep = sess.run(getattr(content_vgg, CONTENT_LAYER), feed_dict={content: content_img})
        style_rep = sess.run(get_style_rep(style_vgg), feed_dict={style: style_img})

        noise = tf.random.truncated_normal(content_img.shape, stddev=0.1 * np.std(content_img))
        image = tf.Variable(noise)
        image_vgg = vgg19.Vgg19(vgg_path)
        image_vgg.build(image)

        content_loss = tf.nn.l2_loss(getattr(image_vgg, CONTENT_LAYER) - content_rep) / content_rep.size
        style_loss = compute_style_loss(style_rep, image_vgg)
        loss = ALPHA * content_loss + BETA * style_loss
        optimizer = tf.compat.v1.train.AdamOptimizer(LR).minimize(loss)

        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(1, iterations + 1):
            sess.run(optimizer)
            print(f"Iteration {i}/{iterations}, Content Loss: {sess.run(content_loss)}, Style Loss: {sess.run(style_loss)}")

            # output_path = os.path.join(output_dir, f'output_{i:04}.jpg')
            
            if i == iterations:
                save_image(sess.run(image), output_path, content_yuv if preserve_color else None)

if __name__ == '__main__':
    content_dirs = ["content1"]
    content_nums = ["01", "02", "03", "04", "05", "06"]
    # content_nums = ["13", "14", "15", "16", "17", "18", "19"]
    style_dirs = ["style_01", "style_02", "style_03", "style_04", "style_05", "style_06"]
    # style_dirs = ["style_13", "style_14", "style_15", "style_16", "style_17", "style_18"]
    preprocess_types = ["pca"]
    # preprocess_types = ["origin", "lab", "luv", "match_hist", "mean_std", "pca", "pdf"]
    path = os.path.abspath(os.getcwd())
    
    for content_dir in content_dirs:
        for content_num in content_nums:
            for style_dir in style_dirs:
                content_image_path = os.path.join(path, f'data/Content/{content_dir}/{content_num}.jpg')
                style_image_path = os.path.join(path, f'transfered_style/{content_dir}/{content_num}/{style_dir}/origin.jpg')
                os.makedirs(f"result/{content_dir}/{content_num}/{style_dir}", exist_ok=True)
                output_path = f"result/{content_dir}/{content_num}/{style_dir}/preserve_color.jpg"
                main(content_image_path, style_image_path, output_path, 800, 'vgg19.npy', 'preserve_color')
                for preprocess_type in preprocess_types:
                    style_image_path = os.path.join(path, f'transfered_style/{content_dir}/{content_num}/{style_dir}/{preprocess_type}.jpg')
                    output_path = f"result/{content_dir}/{content_num}/{style_dir}/{preprocess_type}.jpg"
                    main(content_image_path, style_image_path, output_path, 800, 'vgg19.npy', None)
