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


def test_hdvila_dataset_and_dataloader(csv_path, batch_size=2, num_workers=2):
    # 创建 HyperStageDataset 实例
    dataset = HyperStageDataset(
        csv_path=csv_path,
        width=512,
        height=512,
        n_sample_frames=8,
        sample_frame_rate=2
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # 测试数据集大小
    print(f"Dataset size: {len(dataset)}")

    # 遍历数据加载器并检查几个批次
    for i, batch in enumerate(dataloader):
        if i >= 3:  # 只检查前3个批次
            break

        print(f"\nBatch {i + 1}:")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        print(f"Pose shape: {batch['pose'].shape}")
        print(f"Sentence length: {len(batch['sentence'][0])}")

        # 检查数据类型和值范围
        assert isinstance(batch['pixel_values'], torch.Tensor), "pixel_values should be a torch.Tensor"
        assert isinstance(batch['pose'], torch.Tensor), "pose should be a torch.Tensor"
        assert batch['pixel_values'].shape == (batch_size, 8, 3, 512, 512), "Unexpected shape for pixel_values"
        assert batch['pose'].shape == (batch_size, 8, 3, 512, 512), "Unexpected shape for pose"
        assert -1.0 <= batch['pixel_values'].min() <= batch['pixel_values'].max() <= 1.0, "pixel_values should be in range [-1, 1]"
        assert -1.0 <= batch['pose'].min() <= batch['pose'].max() <= 1.0, "pose should be in range [-1, 1]"
        assert len(batch['sentence']) == batch_size, "Unexpected number of sentences"

        # 打印第一个样本的一些信息
        print(f"First sentence: {batch['sentence'][0][:50]}...")  # 只打印前50个字符

    save_image(batch['pixel_values'][0] + 1, 'test.png')
    save_image(batch['pose'][0] + 1, 'test_pose.png')
    print("\nDataLoader test completed successfully!")

if __name__ == "__main__":
    # 请替换为你的实际 CSV 文件路径
    csv_path = "/home/sora/workspace/meta_caption.csv"
    test_hdvila_dataset_and_dataloader(csv_path)