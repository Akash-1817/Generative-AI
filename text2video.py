from diffusers import DiffusionPipeline
import torch
import numpy as np
import imageio
from PIL import Image

# Load pipeline
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Generate batched video frames
prompt = input('enter a text:')
video_batches = pipe(prompt, num_inference_steps=50).frames

# Frame upscaling target
target_resolution = (512, 512)

# Flatten & process frames
processed_frames = []

for i, batch in enumerate(video_batches):
    for j, frame in enumerate(batch):
        # Normalize and convert to uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # Ensure RGB format
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] > 4:
            frame = frame[:, :, :3]

        # Convert to PIL for resizing
        image = Image.fromarray(frame)
        image = image.resize(target_resolution, Image.LANCZOS)
        processed_frames.append(np.array(image))

print(f"✅ Total processed frames: {len(processed_frames)}")

# Save high-quality MP4 using FFmpeg writer
output_path = "output.mp4"
writer = imageio.get_writer(
    output_path,
    fps=10,
    codec='libx264',          # High quality H.264 codec
    bitrate="10M",             # Increase bitrate for better quality
    quality=10                # Max quality
)

for frame in processed_frames:
    writer.append_data(frame)
writer.close()

print(f"video saved as {output_path}")