from glob import glob
import os

sequences = glob('kitti_tracking/images/*')

for seq in sequences:
    output_video_path = f'kitti_tracking/tracking_images/{seq.split("/")[-1]}.mp4'
    cmd_str = f'ffmpeg -y -framerate 10 -f image2 -i {seq}/%06d.jpg -c:v libx264 -preset fast -x264-params crf=25 -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -r 30 {output_video_path}'
    print(f'Running ffmpeg with cmd:\n{cmd_str}')
    os.system(cmd_str)
