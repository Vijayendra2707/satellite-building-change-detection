
import numpy as np
import torch
from PIL import Image


def load_image(path):
    """
    Load an RGB satellite image.
    """

    image = np.array(
        Image.open(path).convert("RGB")
    )

    return image


def preprocess_patch(patch):
    """
    Convert HWC uint8 image patch
    to normalized CHW tensor.
    """

    patch = (
        patch.astype(np.float32) / 255.0
    )

    patch = torch.from_numpy(
        patch
    ).permute(2, 0, 1)

    return patch


@torch.no_grad()
def predict_full_image(
    model,
    image_a,
    image_b,
    device,
    threshold=0.70,
    patch_size=256
):
    """
    Perform full-resolution change detection.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Siamese change detection model.

    image_a : np.ndarray
        Before image, H x W x 3.

    image_b : np.ndarray
        After image, H x W x 3.

    device : torch.device
        CPU or CUDA device.

    threshold : float
        Probability threshold for binary change mask.

    patch_size : int
        Size of model input patches.

    Returns
    -------
    probability_map : np.ndarray
        H x W probability map.

    change_mask : np.ndarray
        H x W binary change mask.
    """

    if image_a.shape != image_b.shape:
        raise ValueError(
            "Before and after images must have "
            "the same dimensions."
        )

    height, width, channels = image_a.shape

    if channels != 3:
        raise ValueError(
            "Expected RGB images with 3 channels."
        )

    if (
        height % patch_size != 0
        or width % patch_size != 0
    ):
        raise ValueError(
            f"Image dimensions must be divisible "
            f"by patch size {patch_size}."
        )

    model.eval()

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for y in range(
        0,
        height,
        patch_size
    ):

        for x in range(
            0,
            width,
            patch_size
        ):

            patch_a = image_a[
                y:y + patch_size,
                x:x + patch_size
            ]

            patch_b = image_b[
                y:y + patch_size,
                x:x + patch_size
            ]

            tensor_a = preprocess_patch(
                patch_a
            ).unsqueeze(0).to(
                device
            )

            tensor_b = preprocess_patch(
                patch_b
            ).unsqueeze(0).to(
                device
            )

            logits = model(
                tensor_a,
                tensor_b
            )

            probabilities = torch.sigmoid(
                logits
            )

            patch_probability = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            probability_map[
                y:y + patch_size,
                x:x + patch_size
            ] = patch_probability

    change_mask = (
        probability_map >= threshold
    ).astype(np.uint8)

    return (
        probability_map,
        change_mask
    )


def predict_from_paths(
    model,
    before_path,
    after_path,
    device,
    threshold=0.70,
    patch_size=256
):
    """
    Run change detection directly
    from two image paths.
    """

    image_a = load_image(
        before_path
    )

    image_b = load_image(
        after_path
    )

    probability_map, change_mask = (
        predict_full_image(
            model=model,
            image_a=image_a,
            image_b=image_b,
            device=device,
            threshold=threshold,
            patch_size=patch_size
        )
    )

    return (
        image_a,
        image_b,
        probability_map,
        change_mask
    )


def save_change_mask(
    change_mask,
    output_path
):
    """
    Save binary change mask as PNG.
    """

    mask = (
        change_mask * 255
    ).astype(np.uint8)

    Image.fromarray(
        mask
    ).save(
        output_path
    )
