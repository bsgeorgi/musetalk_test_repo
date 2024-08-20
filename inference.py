import argparse
import os
import numpy as np
import cv2
import torch
import glob
import pickle
import json
from tqdm import tqdm
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending
import shutil
import threading
import queue
import time
import subprocess

# Load model weights
audio_processor, vae, unet, positional_encoder = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
positional_encoder = positional_encoder.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

# Default configuration
DEFAULT_VIDEO_PATH = "data/test.mp4"
DEFAULT_AUDIO_CLIPS = {"audio_0": "data/audio/test.mp3"}
DEFAULT_BBOX_SHIFT = 8
DEFAULT_PREPARATION = False

def create_hls_stream(frames, audio_path, output_dir, size, fps, segment_time=10):
    os.makedirs(output_dir, exist_ok=True)

    # Create a named pipe for feeding video frames to FFmpeg
    pipe_path = os.path.join(output_dir, 'video_pipe.y4m')
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)

    # Start FFmpeg process for HLS creation
    ffmpeg_command = [
        'ffmpeg', '-y', '-f', 'yuv4mpegpipe', '-i', pipe_path,
        '-i', audio_path, '-c:v', 'libx264', '-c:a', 'aac',
        '-f', 'hls', '-hls_time', str(segment_time), '-hls_playlist_type', 'vod',
        os.path.join(output_dir, 'index.m3u8')
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_command)

    # Open the pipe and write video frames in YUV4MPEG format
    with open(pipe_path, 'wb') as pipe:
        header = f'YUV4MPEG2 W{size[0]} H{size[1]} F{fps}:1 Ip A0:0 C444\n'.encode()
        pipe.write(header)
        for frame in frames:
            pipe.write(b'FRAME\n')
            pipe.write(frame.tobytes())

    ffmpeg_process.communicate()
    os.remove(pipe_path)

class AvatarInference:
    def __init__(self, avatar_id, video_path=DEFAULT_VIDEO_PATH, bbox_shift=DEFAULT_BBOX_SHIFT, 
                 batch_size=4, preparation=DEFAULT_PREPARATION):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation

        # Paths initialization
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avatar_info.json"

        self._initialize()

    def _initialize(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                shutil.rmtree(self.avatar_path)
                print(f"Re-creating avatar: {self.avatar_id}")
            else:
                print(f"Creating avatar: {self.avatar_id}")
            os.makedirs(self.avatar_path)
            self._prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                raise FileNotFoundError(f"{self.avatar_id} does not exist, set preparation to True")
            self._load_existing_materials()

    def _load_existing_materials(self):
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        self.frame_list_cycle = read_imgs(sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png'))))
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        self.mask_list_cycle = read_imgs(sorted(glob.glob(os.path.join(self.mask_out_path, '*.png'))))

    def _process_frames(self, res_frame_queue, video_len):
        self.final_frames = []
        self.idx = 0
        while self.idx < video_len - 1:
            try:
                res_frame = res_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = self.frame_list_cycle[self.idx % len(self.frame_list_cycle)]
            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]

            res_frame = cv2.resize(res_frame.astype(np.uint8), (bbox[2] - bbox[0], bbox[3] - bbox[1]))
            combined_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            self.final_frames.append(combined_frame)
            self.idx += 1

    def inference(self, audio_path, out_vid_name, fps):
        print("Starting inference...")

        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        print(f"Audio processing took {(time.time() - start_time) * 1000}ms")

        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        process_thread = threading.Thread(target=self._process_frames, args=(res_frame_queue, video_num))
        process_thread.start()

        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()

        for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size))):
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=device, dtype=unet.model.dtype)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)

            audio_feature_batch = positional_encoder(audio_feature_batch)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)

        process_thread.join()

        if out_vid_name:
            height, width, _ = self.final_frames[0].shape
            size = (width, height)

            hls_output_dir = os.path.join(self.video_out_path, f"{out_vid_name}_hls")

            # Create HLS streamable files directly from frames
            create_hls_stream(self.final_frames, audio_path, hls_output_dir, size, fps)

            print(f"HLS stream saved to {hls_output_dir}")

        print(f"Inference completed in {time.time() - start_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default 8)
    parser.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--audio_clips", type=json.loads, default=json.dumps(DEFAULT_AUDIO_CLIPS))
    parser.add_argument("--bbox_shift", type=int, default=DEFAULT_BBOX_SHIFT)
    parser.add_argument("--preparation", type=bool, default=DEFAULT_PREPARATION)

    args = parser.parse_args()

    avatar_inference = AvatarInference(
        avatar_id="Peccy",
        video_path=args.video_path,
        bbox_shift=args.bbox_shift,
        batch_size=args.batch_size,
        preparation=args.preparation
    )

    for audio_num, audio_path in args.audio_clips.items():
        print(f"Inferring using: {audio_path}")
        avatar_inference.inference(audio_path, audio_num, args.fps)
