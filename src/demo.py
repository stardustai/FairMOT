from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import logging
import os
from glob import glob
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import pandas as pd
from tqdm import tqdm
import numpy as np

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, opt.input_video.split('/')[-1]).replace('.mp4', '.txt')
    opt.result_file = result_filename
    frame_rate = dataloader.frame_rate

    # frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame')
    # eval_seq(opt, dataloader, opt.dataset, result_filename,
    #          save_dir=frame_dir, show_image=opt.show_image, frame_rate=frame_rate,
    #          use_cuda=opt.gpus!=[-1])

    # if opt.output_format == 'video':
    #     filename = opt.input_video.split('/')[-1].split('.')
    #     filename = filename[0]+'_result.'+filename[-1]
    #     output_video_path = os.path.join(result_root, filename)
    #     cmd_str = f'ffmpeg -y -framerate {frame_rate} -f image2 -i {os.path.join(result_root, "frame")}/%05d.jpg -c:v libx264 -preset fast -x264-params crf=25 -vf fps={frame_rate} {output_video_path}'
    #     logger.info(f'Running ffmpeg with cmd:\n{cmd_str}')
    #     os.system(cmd_str)

def post_processing(opt):
    results = pd.read_csv(opt.result_file)
    ids = results.id.unique()
    ids.sort()
    frames = results.frame.unique()
    frames.sort()
    # detection is frame x id x 5 (x1, y1, x2, y2, label, score) matrix
    detections = np.zeros((len(frames), len(ids), 5))
    #fill value
    for tid in tqdm(ids):
        tracks = results.query('id == @tid')
        # remove short ids
        if len(tracks) < 30:
            print(f'Skip object [{tid}] with only {len(tracks)} frames')
            continue
        # interpolate between frames -> use
        xp = tracks.frame
        for i, yi in enumerate(['x1', 'y1', 'x2', 'y2']):
            yp = tracks[yi]
            x = range(xp.min(), xp.max())
            y = np.interp(x, xp, yp)
            detections[xp, ]
        # fill back values to detections
        # or even add some smoothing
    


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init(
        [
            '--data_cfg', 'src/lib/cfg/kitti.json',
            # '--load_model', 'models/ctdet_coco_dla_2x.pth',
            '--load_model', 'exp/mot/kitti_c6/model_last.pth',
            '--num_classes', '6',
            '--ltrb', False,
            '--track_buffer', '150',
            '--dataset', 'kitti'
    ])
    opt.input_video = 'videos/video_10s.mp4'
    opt.output_root = 'output'
    opt.show_image = False
    demo(opt)
    post_processing(opt)

    #kitti
    # for v in glob('videos/kitti/*.mp4'):
    #     opt.input_video = v
    #     demo(opt)

    #bosch
    # opt.output_root = 'output/bosch'
    # for v in glob('videos/bosch_tracking/*.mp4'):
    #     opt.input_video = v
    #     demo(opt)
