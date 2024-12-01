# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
from color_transfer import ColorTransfer
import gradio as gr


def transfer_strength(func, img_arr_in, img_arr_ref, transfer_strength):
    """
    Apply color transfer with strength adjustment.

    Args:
        func: Color transfer function from ColorTransfer.
        img_arr_in: Input image as a numpy array (BGR).
        img_arr_ref: Reference image as a numpy array (BGR).
        transfer_strength: Float between 0.0 and 1.0 indicating transfer strength.

    Returns:
        The color-transferred image as a numpy array (BGR).
    """
    # Perform the transfer
    transferred = func(img_arr_in, img_arr_ref)
    # Blend the result with the original image based on strength
    blended = cv2.addWeighted(
        transferred, transfer_strength, img_arr_in, 1 - transfer_strength, 0
    )
    return blended


def process_images(img_in, ref_in, strength):
    """Process the input and reference images using multiple color transfer methods with adjustable strength."""
    img_arr_in = cv2.cvtColor(np.array(img_in), cv2.COLOR_RGB2BGR)
    img_arr_ref = cv2.cvtColor(np.array(ref_in), cv2.COLOR_RGB2BGR)

    CT = ColorTransfer()

    results = {}
    times = []

    # Run each color transfer method with strength adjustment
    methods = {
        "Histogram Transfer": CT.match_histograms,
        "PCA Transfer": CT.pca_transfer,
        "Mean-Std Transfer": CT.mean_std_transfer,
        "Lab Transfer": CT.lab_transfer,
        "Luv Transfer": CT.luv_transfer,
        "PDF Transfer": lambda img, ref: CT.pdf_transfer(img, ref, regrain=False),
    }

    # Initialize the results dictionary
    results = {key: np.zeros_like(img_arr_in) for key in methods}

    for method, func in methods.items():
        try:
            start_time = time.time()
            # Use the helper function to apply strength adjustment
            results[method] = transfer_strength(func, img_arr_in, img_arr_ref, strength)
            # save results image
            # cv2.imwrite(f"temp_{method.replace(' ', '_')}.jpg", results[method])

            results[method] = cv2.cvtColor(results[method], cv2.COLOR_BGR2RGB)
            elapsed = time.time() - start_time
            times.append(f"{method}: {elapsed:.2f}s")
        except Exception as e:
            results[method] = np.zeros_like(img_arr_in)
            times.append(f"{method}: Error ({e})")

        yield (
            np.array(img_in),
            results["Histogram Transfer"],
            results["PCA Transfer"],
            results["Mean-Std Transfer"],
            results["Lab Transfer"],
            results["Luv Transfer"],
            results["PDF Transfer"],
        )


def gradio_interface():
    # Define Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Color Transfer and Style Transfer")
        gr.Markdown(
            "Upload an input image and a reference image to perform color transfer using various methods. "
            "Then apply style transfer (StyleShot) using the color-transferred results."
        )

        with gr.Row():
            input_image = gr.Image(
                type="pil", label="Input Image/Style Image", height=300
            )
            reference_image = gr.Image(
                type="pil", label="Reference Image/Content Image", height=300
            )

        # Color Transfer Section
        gr.Markdown("### Color Transfer with Adjustable Strength")
        strength_slider = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="Transfer Strength"
        )
        # log = gr.Textbox(label="Processing Times", interactive=False)

        with gr.Row():
            output_original = gr.Image(type="numpy", label="Original Image")
            output_mh = gr.Image(type="numpy", label="Histogram")
            output_pca = gr.Image(type="numpy", label="PCA")
            output_mean = gr.Image(type="numpy", label="Mean-Std")
            output_lab = gr.Image(type="numpy", label="Lab")
            output_luv = gr.Image(type="numpy", label="Luv")
            output_pdf = gr.Image(type="numpy", label="PDF")

        # Button to process color transfer
        transfer_button = gr.Button("Run Color Transfer")
        transfer_button.click(
            fn=process_images,
            inputs=[input_image, reference_image, strength_slider],
            outputs=[
                output_original,
                output_mh,
                output_pca,
                output_mean,
                output_lab,
                output_luv,
                output_pdf,
            ],
        )

    return demo


# Launch Gradio interface
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch(share=True)
