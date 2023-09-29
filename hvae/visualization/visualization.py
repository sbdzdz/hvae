"""Visualization utilities."""
import io
from pathlib import Path

import numpy as np
import PIL
from matplotlib import pyplot as plt

repo_root = Path(__file__).parent.parent.parent


def draw_batch(
    images,
    fig_height: float = 10,
    num_images: int = 16,
    normalize: bool = True,
):
    """Show a batch of images on a grid.
    Only the first n_max images are shown.
    Args:
        images: A numpy array of shape (N, C, H, W) or (N, H, W)
        fig_height: The height of the figure in inches
        num_images: The number of images to show
    Returns:
        None
    """
    num_images = min(images.shape[0], num_images)
    if images.ndim == 4:
        images = np.transpose(images, (0, 2, 3, 1))
    ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(
        ncols, nrows, figsize=(ncols / nrows * fig_height, fig_height)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, img in zip(axes.flat, images[:num_images]):
        if normalize:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        ax.imshow(img, cmap="Greys_r", interpolation="nearest")
        ax.axis("off")
    buffer = io.BytesIO()
    plt.savefig(buffer, bbox_inches="tight")
    plt.close()

    return PIL.Image.open(buffer)


def draw_reconstructions(
    *image_arrays,
    fig_height: float = 10,
    num_images: int = 16,
    normalize: bool = True,
):
    """Show a batch of images and their reconstructions on a grid.
    Only the first n_max images are shown.
    Args:
        image_arrays: Numpy arrays of shape (N, C, H, W) or (N, H, W)
        fig_height: The height of the figure in inches
        num_images: The number of images to show
    Returns:
        None
    """
    img = image_arrays[0]
    num_images = min(img.shape[0], num_images)
    if img.ndim == 4:
        image_arrays = [np.transpose(img, (0, 2, 3, 1)) for img in image_arrays]
    ncols = int(np.ceil(np.sqrt(num_images)))
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(len(image_arrays) * ncols / nrows * fig_height, fig_height),
    )
    fig.tight_layout()
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax in axes.flat:
        ax.axis("off")

    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        if normalize:
            images = [
                (img[i] - np.min(img[i])) / (np.max(img[i]) - np.min(img[i]))
                for img in image_arrays
            ]
        concatenated = np.concatenate(images, axis=1)
        border_width = concatenated.shape[1] // 128 or 1

        for j in range(1, len(image_arrays)):
            mid = j * concatenated.shape[1] // len(image_arrays)
            concatenated[:, mid - border_width : mid + border_width] = 1.0
        ax.imshow(concatenated, cmap="Greys_r", interpolation="nearest")

    buffer = io.BytesIO()
    plt.savefig(buffer, bbox_inches="tight")
    plt.close()

    return PIL.Image.open(buffer)
