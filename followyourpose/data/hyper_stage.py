import os
import csv
import random
from functools import lru_cache
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms._transforms_video as transforms_video
from torchvision.utils import save_image
from decord import VideoReader, cpu

class HyperStageDataset(Dataset):
    def __init__(self,
                 csv_path,
                 width=512,
                 height=512,
                 n_sample_frames=8,
                 sample_frame_rate=2):
        self.csv_path = csv_path
        self.resolution = (height, width)
        self.video_length = n_sample_frames
        self.frame_stride = sample_frame_rate
        
        self._load_metadata()
        self._setup_transforms()

    def _load_metadata(self):
        self.metadata = []
        with open(self.csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.metadata.append({
                    'path': row['path'],
                    'text': row['text'],
                    'pose_video': row['pose_video']
                })
        print(f'Loaded {len(self.metadata)} items from CSV')

    def _setup_transforms(self):
        self.spatial_transform = transforms.Compose([
            transforms.Resize(self.resolution[0]),
            transforms_video.CenterCropVideo(self.resolution[0]),
        ])

    def _load_video(self, video_path):
        return VideoReader(video_path, ctx=cpu(0))

    def _sample_frame_indices(self, video_length):
        if self.frame_stride != 1:
            all_frames = list(range(0, video_length, self.frame_stride))
            if len(all_frames) < self.video_length:
                fs = max(1, video_length // self.video_length)
                all_frames = list(range(0, video_length, fs))
        else:
            all_frames = list(range(video_length))
        
        if len(all_frames) <= self.video_length:
            return all_frames + [all_frames[-1]] * (self.video_length - len(all_frames))
        
        start = random.randint(0, len(all_frames) - self.video_length)
        return all_frames[start:start + self.video_length]

    def __getitem__(self, index):
        item = self.metadata[index]
        video_path = item['path']
        pose_video_path = item['pose_video']

        try:
            video_reader = self._load_video(video_path)
            pose_video_reader = self._load_video(pose_video_path)

            frame_indices = self._sample_frame_indices(len(video_reader))
            
            frames = video_reader.get_batch(frame_indices).asnumpy()
            pose_frames = pose_video_reader.get_batch(frame_indices).asnumpy()

            frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
            pose_frames = torch.from_numpy(pose_frames).permute(3, 0, 1, 2).float()

            frames = self.spatial_transform(frames)
            pose_frames = self.spatial_transform(pose_frames)

            frames = (frames / 127.5 - 1.0).permute(1, 0, 2, 3)
            pose_frames = (pose_frames / 225).permute(1, 0, 2, 3)

            return {
                'pixel_values': frames,
                'sentence': item['text'],
                'pose': pose_frames
            }
        except Exception as e:
            print(f"Error loading video {video_path} or pose video {pose_video_path}: {str(e)}")
            # Return a default or placeholder item
            return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return len(self.metadata)
