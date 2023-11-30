# Endo(morphism)Zoom

Creates a video that zoom into itself using an outpainting model

## Usage
```
Usage: main.py [OPTIONS] VIDEO_OUTPUT_FILE

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
  --help                          Show this message and exit.

```

## Example

```
sh# python main.py \
 --prompt "A sprawling underwater city with transparent domes and tunnels, surrounded by colorful coral reefs and schools of luminous fish. Residents, diverse in descent, are seen traveling in bubble-like submarines" \
 --fps 30 \
 --iterations 30 \
 output.mp4
```

[![Underwater City Zoom](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DR8WQQYMIcqA)](https://www.youtube.com/watch?v=R8WQQYMIcqA)

## Tips

Monitor the temp directory while running to look at the images that will be used for the zoom


