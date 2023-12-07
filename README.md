# Endo(morphism)Zoom

Creates a video that zoom into itself using an outpainting model

## Usage
```

Usage: python main.py [OPTIONS] VIDEO_OUTPUT_FILE

Options:
  --height INTEGER                Height of the output image  [default: 512]
  --width INTEGER                 Width of the output image.  [default: 896]
  --prompt TEXT                   Prompt to use.
  --iterations INTEGER            Number of images to loop between.  [default:
                                  20]
  --scaling INTEGER               How much to scale down an image before
                                  outpainting.  [default: 40]
  --frames-per-iterations INTEGER
                                  Number of frames to generate per iteration.
                                  [default: 120]
  --fps INTEGER                   Frames per second.  [default: 30]
  --stable-diffusion-api-key TEXT
                                  API key for the stable diffiusion api.
                                  Should be set using the environment variable
                                  STABILITY_API_KEY for security
  --input-image PATH_TO_IMAGE     Provide a seed image instead of using an AI-generated one
  --help                          Show this message and exit.

```

## Example

Fully Generated Image
```
sh# python main.py \
 --prompt "A sprawling underwater city with transparent domes and tunnels, surrounded by colorful coral reefs and schools of luminous fish. Residents, diverse in descent, are seen traveling in bubble-like submarines" \
 --fps 30 \
 --iterations 30 \
 output.mp4
```

Provide an initial image
```
sh# python main.py \
 --prompt "A dystopian city surrounded by desert. Cats are the only living creatures seen roaming the land. In the style of anime water painting" \
 --fps 30 \
 --iterations 30 \
 --initial-image images/ai-image.png
 output.mp4
```

Notes on input image size: the SDXL v2.2.2 API we use have the following requirements
- Width and Height MUST be above 128
- Only ONE height or width may be above 512 (512x768 is valid but 578x768 is not)
- Maximum dimensions supported are 512x896 or 896x512

[![Underwater City Zoom](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DR8WQQYMIcqA)](https://www.youtube.com/watch?v=R8WQQYMIcqA)

## Tips

Monitor the temp directory while running to look at the images that will be used for the zoom
