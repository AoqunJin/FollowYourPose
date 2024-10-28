import csv
import cv2
import os

def extract_frames_from_video(video_path, output_dir, text, task_name, frame_interval=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video information
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = frame_count / fps
    
    frame_number = 0
    extracted_frame_number = 0
    output_data = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.listdir(output_dir):
        extracted_frame_number = len(os.listdir(output_dir))
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        if model == "first_5_frames":
            frame_interval = 1
            if frame_number > 4:
                break
        if model == "last_2_frames":
            frame_interval = 1
            if frame_number < frame_count - 2:
                frame_number += 1
                continue
        if model == "last_1_frames":
            frame_interval = 1
            if frame_number < frame_count - 1:
                frame_number += 1
                continue
        
        # Save frames at specified interval (every nth frame)
        if (frame_number + 1) % frame_interval == 0:
            output_file = os.path.join(output_dir, f"{task_name}_{extracted_frame_number}.png")
            cv2.imwrite(output_file, frame)
            output_data.append({
                'image': output_file,
                'text': text,
                # 'task': task_name
            })
            extracted_frame_number += 1
        
        frame_number += 1
    
    cap.release()
    return output_data

def process_csv_and_split_videos(csv_path, output_dir, output_csv_path, frame_interval=30):
    output_rows = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            video_path = row['path']
            text = row['text']
            
            # Extract the task name from the video file name
            task_name = os.path.basename(video_path).split('_')[1:-2]
            task_name = '_'.join(task_name) if task_name else 'unknown_task'
            
            # Define the directory where frames will be saved
            video_output_dir = os.path.join(output_dir, task_name)
            
            print(f"Processing video: {video_path} with task: {task_name}")
            
            # Extract frames and save them with the associated text
            extracted_frames = extract_frames_from_video(video_path, video_output_dir, text, task_name, frame_interval)
            
            # Collect rows for CSV
            output_rows.extend(extracted_frames)

    # Write the output CSV
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['image', 'text'])
        writer.writeheader()
        writer.writerows(output_rows)

# Example usage
csv_path = '/home/ao/workspace/fs/diffusers/meta_caption_random.csv'
output_dir = '/home/ao/workspace/fs/diffusers/hf_data_rl_2'
output_csv_path = '/home/ao/workspace/fs/diffusers/metadata.csv'
frame_interval = 100  # Extract every 3th frame

# first_5_frames | last_2_frames | all_frames
model = "all_frames"
process_csv_and_split_videos(csv_path, output_dir, output_csv_path, frame_interval)
