from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=opt.show_image, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        filename = opt.input_video.split('/')[-1].split('.')
        filename = filename[0]+'_result.'+filename[-1]
        output_video_path = osp.join(result_root, filename)
        cmd_str = f'ffmpeg -f image2 -i {osp.join(result_root, "frame")}/%05d.jpg -c:v libx265 -preset fast -x265-params crf=30 -tag:v hvc1 -vf fps={frame_rate} {output_video_path}'
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    opt.load_model = 'models/fairmot_dla34.pth'
    # opt.device = 'cpu'
    opt.input_video = 'videos/video_10s.mp4'
    opt.show_image = True
    opt.output_root = 'output'
    demo(opt)
