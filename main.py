import os
import sys

from dataclasses import dataclass
from pathlib import Path
import click
import shutil
import numpy as np
import requests
import base64
import cv2

@dataclass
class Image:
    width: int
    height: int
    path: Path = Path("temp/init.png")  # Default value for path

    def ratio(self) -> float:
        return self.width/self.height

    @classmethod
    def create_default_image(cls):
        return cls(width=896, height=512)  # Set default values for width and height as needed


engine_id = "stable-diffusion-xl-beta-v2-2-2"
upscale_engine_id = "esrgan-v1-x2plus"

api_host = os.getenv('API_HOST', 'https://api.stability.ai')

def get_new_image_from_ai_generator(prompt: str, image: Image, api_key, sampler="K_EULER", steps=40):
    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": 8,
            "sampler": "K_EULER",
            "height": image.height,
            "width": image.width,
            "samples": 1,
            "steps": steps
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    with open(image.path, "wb") as f:
        image_data = base64.b64decode(data["artifacts"][0]["base64"])

        f.write(image_data)

def resize_image(img: np.ndarray, scale_percent: float) -> (np.ndarray, (int, int)):
    # get the original dimensions
    width = img.shape[1]
    height = img.shape[0]

    # calculate the new dimensions
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)

    # resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)

    return resized_img

# Add outpainting to an image
def edit_image(prompt, input_image, mask_image, output_image, api_key, sampler="K_EULER", steps=40):
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image/masking",
        headers={
            "Accept": 'application/json',
            "Authorization": f"Bearer {api_key}"
        },
        files={
            'init_image': open(input_image, 'rb'),
            'mask_image': open(mask_image, 'rb'),
        },
        data={
            "mask_source": "MASK_IMAGE_WHITE",
            "text_prompts[0][text]": prompt,
            "cfg_scale": 7,
            "sampler": sampler,
            "clip_guidance_preset": "SLOW",
            "samples": 1,
            "steps": steps
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    with open(output_image, "wb") as f:
        image_data = base64.b64decode(data["artifacts"][0]["base64"])

        f.write(image_data)


def upscale(input_image: Path, output_image: Path, new_width, api_key):
    '''Upscale an image using stable diffusion api (can also be done locally)'''

    response = requests.post(
        f"{api_host}/v1/generation/{upscale_engine_id}/image-to-image/upscale",
        headers={
            "Accept": "image/png",
            "Authorization": f"Bearer {api_key}"
        },
        files={
            "image": open(input_image, "rb")
        },
        data={
            "width": new_width,
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    with open(output_image, "wb") as f:
        f.write(response.content)


# TODO: This function does WAY to much. Split it up into smaller functions
def insert_center_and_mask(img: np.ndarray, final_image: Image) -> (np.ndarray, np.ndarray, float, float):
    # get dimensions of image
    height, width, _ = img.shape
    ratio = width / height

    # get center point of final image
    final_image.center = (final_image.height // 2, final_image.width // 2)

    # Create a mask with correct size
    new_img = np.zeros((final_image.height, final_image.width, 3), dtype=np.uint8)
    mask = np.full((final_image.height, final_image.width, 3), (255, 255, 255), dtype=np.uint8)

    # Add the scaled image to the new image (with black background)
    start_y = (final_image.height - height) // 2
    start_x = (final_image.width - width) // 2
    new_img[start_y:start_y + height, start_x:start_x + width] = img

    # Draw the mask ellipse used for outpainting
    ax = int(width * 0.5)
    by = int(height * 0.5)

    # DEBUG
    # mask[start_y:start_y+height, start_x:start_x+width] = img

    mask = cv2.ellipse(img = mask,
                       center=final_image.center,
                       axes=(ax, by),
                       angle = 0,
                       startAngle=0,
                       endAngle=360,
                       color=(0, 0, 0),
                       thickness=-1)


    # Calculate the zoom factor by calculating the largest possible rectangle that can
    # fit inside the mask ellipse. Geomatry and shit
    height_prime = int(2 * np.sqrt((ax**2 * by**2) / (by**2 * ratio**2 + ax**2)))
    width_prime = int(ratio * height_prime)

    zoom_factor_start = 1 / (height_prime / final_image.height)
    zoom_factor_end = 1 / (height_prime / height)
    zoom_factor_request = 1 / (height / final_image.height)

    print (f"- Zoom factor start: {zoom_factor_start}, end: {zoom_factor_end}, request: {zoom_factor_request}")

    # Feather the mask for a smoother outpainting
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return new_img, mask, zoom_factor_start, zoom_factor_end

def get_upscale_image_max_width(image: Image) -> int:
    '''Return the maximum upscale width that does not exceed the API's output image pixel limit'''
    UPSCALE_IMAGE_OUTPUT_LIMIT = 4194304  # Defined in https://platform.stability.ai/docs/features/image-upscaling

    im = cv2.imread(str(image.path))
    h, w, _ = im.shape

    upscale_width = 2560                  # Start with this width optimistically and scale down
    upscale_height = (upscale_width * h) / w

    while upscale_width * upscale_height > UPSCALE_IMAGE_OUTPUT_LIMIT:
        upscale_width *= 0.8              # scale width down 20% and try again
        upscale_height = (upscale_width * h) / w

    return upscale_width


def generate_images(init_image: Image, prompt: str, iterations: int, scaling: int, api_key: str, temp_dir=Path("output")):
    '''Generate a series of images based on a provided seed image'''
    images = []
    zoom_factor_start, zoom_factor_end = None, None

    upscale_width = get_upscale_image_max_width(init_image)
    init_image_upscaled = temp_dir / "init_upscaled.png"
    print("Upscaling initial image...")
    upscale(init_image.path, init_image_upscaled, upscale_width, api_key)

    images.append(init_image_upscaled)

    for i in range(iterations):
        print("*"*80)
        print(f"Processing image {i+1}/{iterations}")
        print("+ Scaling and creating mask")
        resized_image = resize_image(cv2.imread(str(init_image.path)), scaling)
        resized_image, mask_image, zoom_factor_start, zoom_factor_end = insert_center_and_mask(resized_image, init_image)

        mask_image_name = temp_dir / f"frame_{i}_mask.png"
        resized_image_name = temp_dir / f"frame_{i}_resized.png"

        cv2.imwrite(str(mask_image_name), mask_image)
        cv2.imwrite(str(resized_image_name), resized_image)

        print("+ Adding outpainting")
        output_image_name = temp_dir / f"frame_{i}.png"
        output_image_name_upscaled = temp_dir / f"frame_{i}_upscaled.png"
        edit_image(prompt, resized_image_name, mask_image_name, output_image_name, api_key)

        print("+ Upscaling")
        upscale(output_image_name, output_image_name_upscaled, upscale_width, api_key)

        images.append(output_image_name_upscaled)
        init_image.path = output_image_name

    return images, zoom_factor_start, zoom_factor_end

# TODO: Place the previous image into the zoomed image for better resolution in the transition
def crop_and_zoom(image: np.ndarray, factor: float, resize):

    # get the original dimensions
    height, width = image.shape[:2]

    resize_image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_LINEAR)
    cropped_image = resize_image[
        (resize_image.shape[0] - height) // 2:(resize_image.shape[0] - height) // 2 + height,
        (resize_image.shape[1] - width) // 2:(resize_image.shape[1] - width) // 2 + width
    ]

    cropped_image = cv2.resize(cropped_image, resize, interpolation = cv2.INTER_LINEAR)

    return cropped_image

@click.command()
@click.option("--height", show_default=True, default=512, help="Height of the output video")
@click.option("--width", show_default=True, default=896, help="Width of the output video.")
@click.option("--prompt", default="A sprawling underwater city with transparent domes and tunnels, surrounded by colorful coral reefs and schools of luminous fish. Residents, diverse in descent, are seen traveling in bubble-like submarines", help="Prompt to use.")
@click.option("--iterations", show_default=True, default=20, help="Number of images to loop between.")
@click.option("--scaling", show_default=True, default=40, help="How much to scale down an image before outpainting.")
@click.option("--frames-per-iterations", show_default=True, default=120, help="Number of frames to generate per image.")
@click.option("--fps", show_default=True, default=30, help="Frames per second in video.")
@click.option("--stable-diffusion-api-key", envvar="STABILITY_API_KEY", help="API key for the stable diffiusion api. Should be set using the environment variable STABILITY_API_KEY for security")
@click.option("--initial-image", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False, help="Input image for use as the seed of video generation.")
@click.argument("video-output-file", type=click.Path(exists=False, writable=True, dir_okay=False))
def main(height: int, width: int, prompt: str, iterations: int, frames_per_iterations: int, fps: int,
         scaling: int, stable_diffusion_api_key: str, initial_image: str, video_output_file: str):

    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    image = Image(width, height)

    # Determine whether to use provided images or generated image from a prompt
    if initial_image:
        init_image = Path(initial_image)
        if not init_image.exists():
            raise Exception(f"Provided input image not found on {init_image} !")
        shutil.copy(init_image, image.path)

    else:
        print("Generating initial image from prompt...")
        get_new_image_from_ai_generator(prompt, image, stable_diffusion_api_key)

    images, zoom_factor_start, zoom_factor_end = generate_images(image, prompt, iterations, scaling,
                                                                stable_diffusion_api_key, temp_dir)

    # Initiate video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_output_file, fourcc, fps, (image.width, image.height))
    for i, image_name in enumerate(images):
        print(f"{i+1/len(images)}: Processing image {image_name}.")

        img = cv2.imread(str(image_name))

        for frame in range(frames_per_iterations):
            progress = frame / frames_per_iterations

            # Linear scaling/factor does not create a realistic zoom as a constant change on a small image
            # is visible faster. We therefore need an inverted quadratic zoom factor
            zoom_factor = zoom_factor_start - ((1 - (1 - progress) ** 2) * (zoom_factor_start - 1))

            # This is not ideal. We should instead modify the loop to go between (zoom_factor_start, zoom_factor_end)
            if zoom_factor < zoom_factor_end:
                sys.stdout.write("\n")
                break

            output = crop_and_zoom(img, zoom_factor, (image.width, image.height))
            sys.stdout.write(".")
            sys.stdout.flush()

            # DEBUG
            #cv2.imwrite(f"output_m/image_{i}_{frame}.png", output)

            out.write(output)

    out.release()


if __name__ == "__main__":
    main()
