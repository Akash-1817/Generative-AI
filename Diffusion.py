pip install diffusers transformers accelerate torch safetensors opencv-python

from diffusers import StableDiffusionPipeline
import torch
import os
import cv2

# -------------------------------------
# ✅ Load Free Anime Model (No Login Needed)
# -------------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to("cuda")

# -------------------------------------
# 🔹 Output Directory for Frames
# -------------------------------------
output_dir = "anime_frames"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------
# 🔹 Prompt & Generation Settings
# -------------------------------------
prompt = "a cinematic anime landscape with a girl in the wind, cherry blossoms, ultra detailed, by Makoto Shinkai"
num_frames = 10
guidance_scale = 7.5
height, width = 512, 512

# -------------------------------------
# 🔹 Generate Frames
# -------------------------------------
print("🎨 Generating anime-style frames...")
for i in range(num_frames):
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=30,
        generator=torch.manual_seed(i)
    ).images[0]

    image.save(f"{output_dir}/frame_{i:04d}.png")
    print(f"✅ Frame {i+1}/{num_frames} saved")

# -------------------------------------
# 🔹 Create Video from Frames
# -------------------------------------
def create_video_from_frames(folder, output_path, fps=10):
    print("🎞️ Stitching video...")
    images = sorted([img for img in os.listdir(folder) if img.endswith(".png")])
    frame_array = [cv2.imread(os.path.join(folder, img)) for img in images]

    height, width, _ = frame_array[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frame_array:
        out.write(frame)
    out.release()
    print(f"🎬 Done! Video saved as {output_path}")

create_video_from_frames(output_dir, "anime_video.mp4", fps=10)

from diffusers import StableDiffusionPipeline
import torch
import os
import cv2

# ------------------------
# ✨ Settings
# ------------------------

# Model settings
model_name = "CompVis/stable-diffusion-v1-4"

# Prompt settings
prompt = "a japanese swordsman with 2 swords fighting against kratos , first they stand at distance , next frame their eyes gets in fight (closeup) , next they fight until one dies"

# Frame generation settings
num_frames = 10
guidance_scale = 7.5
height, width = 512, 512

# Output folders
frames_output_dir = "anime_frames"
video_output_dir = "anime_videos"
video_filename = "my_anime_video.mp4"

# Frame to video settings
frame_static_duration = 2  # seconds each frame stays (easy to change like a prompt!)
fps = 10  # frames per second for video smoothness

# Random seed for each frame (to make different frames)
manual_seed_start = 0

# ------------------------
# 🎨 Load Diffusion Model
# ------------------------

print("🔵 Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to("cuda")

# ------------------------
# 🎨 Generate Frames
# ------------------------

os.makedirs(frames_output_dir, exist_ok=True)

print("🎨 Generating anime-style frames...")
for i in range(num_frames):
    generator = torch.manual_seed(manual_seed_start + i)

    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=30,
        generator=generator
    ).images[0]

    frame_path = os.path.join(frames_output_dir, f"frame_{i:04d}.png")
    image.save(frame_path)
    print(f"✅ Frame {i+1}/{num_frames} saved: {frame_path}")

# ------------------------
# 🎥 Create Video from Frames
# ------------------------

def create_video_static_duration(
    frame_folder,
    video_output_folder,
    video_name,
    frame_duration_sec,
    fps
):
    print("🎞️ Stitching video with static frame duration...")

    os.makedirs(video_output_folder, exist_ok=True)

    images = sorted([img for img in os.listdir(frame_folder) if img.endswith(".png")])
    frame_array = [cv2.imread(os.path.join(frame_folder, img)) for img in images]

    if not frame_array:
        print("❌ No frames found!")
        return

    height, width, _ = frame_array[0].shape
    output_path = os.path.join(video_output_folder, video_name)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    repeat_count = max(1, int(frame_duration_sec * fps))
    print(f"⏳ Each frame will be repeated {repeat_count} times for {frame_duration_sec} seconds")

    for frame in frame_array:
        for _ in range(repeat_count):
            out.write(frame)

    out.release()
    print(f"🎬 Done! Video saved at {output_path}")

    try:
        from IPython.display import FileLink, display
        display(FileLink(output_path))
    except ImportError:
        print("📂 Video created successfully.")

# ------------------------
# 🛠️ Call the Function to Make Video
# ------------------------

create_video_static_duration(
    frame_folder=frames_output_dir,
    video_output_folder=video_output_dir,
    video_name=video_filename,
    frame_duration_sec=frame_static_duration,
    fps=fps
)

print("🏁 All Done!")
