import os
import sys

from pathlib import Path
import click
import numpy as np
import requests
import base64
import cv2

engine_id = "stable-diffusion-xl-beta-v2-2-2"
upscale_engine_id = "esrgan-v1-x2plus"

api_host = os.getenv('API_HOST', 'https://api.stability.ai')

def new_image(prompt: str, image_name: Path, heigh: int, width: int, api_key, sampler="K_EULER", steps=40):
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
            "height": heigh,
            "width": width,
            "samples": 1,
            "steps": steps
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    with open(image_name, "wb") as f:
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


# Upscale an image using stable diffiusion api (can also be done locally)
def upscale(input_image: Path, output_image: Path, new_width, api_key):
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

    with open(output_image, "wb") as f:
        f.write(response.content)


# TODO: This function does WAY to much. Split it up into smaller functions
def insert_center_and_mask(img: np.ndarray, final_image: (int, int)) -> (np.ndarray, np.ndarray, float, float):
    final_image_height, final_image_width = final_image
    final_image_center = (final_image_height // 2, final_image_width // 2)

    # get diemensions of image
    height, width = img.shape[:2]
    ratio = width / height

    # Create a mask with correct size
    new_img = np.zeros((final_image_height, final_image_width, 3), dtype=np.uint8)
    mask = np.full((final_image_height, final_image_width, 3), (255, 255, 255), dtype=np.uint8)

    # Add the scaled image to the new image (with black background)
    start_y = (final_image_height - height) // 2
    start_x = (final_image_width - width) // 2
    new_img[start_y:start_y+height, start_x:start_x+width] = img

    # Draw the mask ellipse used for outpainting
    ax = int(width * 0.5)
    by = int(height * 0.5)

    # DEBUG
    # mask[start_y:start_y+height, start_x:start_x+width] = img

    mask = cv2.ellipse(img = mask,
                       center=(final_image_center[1], final_image_center[0]),
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

    zoom_factor_start = 1 / (height_prime / final_image_height)
    zoom_factor_end = 1 / (height_prime / height)
    zoom_factor_request = 1 / (height / final_image_height)

    print (f"- Zoom factor start: {zoom_factor_start}, end: {zoom_factor_end}, request: {zoom_factor_request}")

    # Feather the mask for a smoother outpainting
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return new_img, mask, zoom_factor_start, zoom_factor_end


def generate_images(prompt, iterations, scaling, size, api_key, temp_dir=Path("output")):
    images = []
    zoom_factor_start, zoom_factor_end = None, None

    # Calcualte upscale size
    ratio = size[1] / size[0]
    upscale_width = 2560

    print("Generating initial image...")
    init_image = temp_dir / "init.png"
    init_image_upscaled = temp_dir / "init_upscaled.png"
    new_image(prompt, init_image, size[0], size[1], api_key)

    print("Upscaling initial image...")
    upscale(init_image, init_image_upscaled, upscale_width, api_key)

    images.append(init_image_upscaled)

    for i in range(iterations):
        print("*"*80)
        print(f"Processing image {i+1}/{iterations}")
        print("+ Scaling and creating mask")
        resized_image = resize_image(cv2.imread(str(init_image)), scaling)
        resized_image, mask_image, zoom_factor_start, zoom_factor_end = insert_center_and_mask(resized_image, size)

        mask_image_name = temp_dir / f"frame_{i}_mask.png"
        resized_image_name = temp_dir / f"frame_{i}_resized.png"

        cv2.imwrite(str(mask_image_name), mask_image)
        cv2.imwrite(str(resized_image_name), resized_image)

        print("+ Adding outpainting")
        output_image_name = temp_dir / f"frame_{i}.png"
        output_image_name_upscaled = temp_dir / f"frame_{i}_upscaled.png"
        edit_image(prompt, resized_image_name, mask_image_name, output_image_name, edit_image)

        print("+ Upscaling")
        upscale(output_image_name, output_image_name_upscaled, upscale_width)

        images.append(output_image_name_upscaled)
        init_image = output_image_name

    return images, zoom_factor_start, zoom_factor_end


# TODO: Plase the previosu image into the zoomed image for better resolution in the transition
def crop_and_zoom(image, factor, resize):

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
@click.argument("video-output-file", type=click.Path(exists=False, writable=True, dir_okay=False))
def main(height: int, width: int, prompt: str, iterations: int, frames_per_iterations: int, fps: int,
         scaling: int, stable_diffusion_api_key: str, video_output_file: str):

    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_size = (height, width)

    images, zoom_factor_start, zoom_factor_end = generate_images(prompt, iterations, scaling, image_size,
                                                                 stable_diffusion_api_key, temp_dir)

    print(f"Images generates: {len(images)}, Zoom factor start: {zoom_factor_start}, end: {zoom_factor_end}")


    # Initiate video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_output_file, fourcc, fps, (image_size[1], image_size[0]))

    for i, image_name in enumerate(images):
        print(f"{i+1/len(images)}: Processing image {image_name}.")

        image = cv2.imread(str(image_name))

        for frame in range(frames_per_iterations):
            progress = frame / frames_per_iterations

            # Linear scaling/factor does not create a realstic zoom as a contant change on a small image
            # is visible fastr. We therefor need a inverted quadratic zoom factor
            zoom_factor = zoom_factor_start - ((1 - (1 - progress) ** 2) * (zoom_factor_start - 1))

            # This is not ideal. We should instead modify the loop to go between (zoom_factor_start, zoom_factor_end)
            if zoom_factor < zoom_factor_end:
                sys.stdout.write("\n")
                break

            output = crop_and_zoom(image, zoom_factor, (image_size[1], image_size[0]))
            sys.stdout.write(".")
            sys.stdout.flush()

            # DEBUG
            #cv2.imwrite(f"output_m/image_{i}_{frame}.png", output)

            out.write(output)

    out.release()


if __name__ == "__main__":
    main()
