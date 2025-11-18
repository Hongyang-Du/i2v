import os
import cv2

def extract_first_frames(video_folder, output_folder, exts=(".mp4", ".mov", ".avi", ".mkv")):
    """
    Extract the first frame of each video in video_folder
    and save them into output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    videos = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(exts)
    ])

    print(f"Found {len(videos)} videos.")

    for filename in videos:
        video_path = os.path.join(video_folder, filename)

        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        if not success:
            print(f"Failed to read video: {filename}")
            continue

        # save as jpg
        frame_name = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_folder, frame_name)

        cv2.imwrite(output_path, frame)
        print(f"Saved first frame → {output_path}")

        cap.release()


if __name__ == "__main__":
    video_folder = "realestate"        # TODO: 改成你的视频目录
    output_folder = "first_frames" # TODO: 改成你想输出的目录

    extract_first_frames(video_folder, output_folder)
