import pandas as pd
from glob import glob
import os, random, time
from PIL import Image, ImageDraw
from tqdm import tqdm

''' Target format
Generate one txt label file for one image. Each line of the txt label file represents one object. 
The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". 
You can modify src/gen_labels_16.py to generate label files for your custom dataset.

Caltech
   |——————images
   |        └——————00001.jpg
   |        |—————— ...
   |        └——————0000N.jpg
   └——————labels_with_ids
            └——————00001.txt
            |—————— ...
            └——————0000N.txt

In the annotation text, each line is describing a bounding box and has the following format:
[class] [identity] [x_center] [y_center] [width] [height]
The field [class] should be 0. Only single-class multi-object tracking is supported in this version.

The field [identity] is an integer from 0 to num_identities - 1, or -1 if this box has no identity annotation.

*Note that the values of [x_center] [y_center] [width] [height] are normalized by the width/height of the image, so they are floating point numbers ranging from 0 to 1.
'''

''' Source format
    #Values    Name      Description
    ----------------------------------------------------------------------------
    1    frame        Frame within the sequence where the object appearers
    1    track id     Unique tracking id of this object within this sequence
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
    1    truncated    Integer (0,1,2) indicating the level of truncation.
                        Note that this is in contrast to the object detection
                        benchmark where truncation is a float in [0,1].
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.
    
    Sample:
    0 -1 DontCare -1 -1 -10.000000 219.310000 188.490000 245.500000 218.560000 -1000.000000 -1000.000000 -1000.000000 -10.000000 -1.000000 -1.000000 -1.000000
    '''

data_root = 'kitti_tracking'
ori_img_path = data_root+'/tracking_images/training/image_02'
target_img_path = data_root+'/images'
target_txt_path = data_root+'/labels_with_ids'
ori_label_path = data_root+'/original_label/label_02'
class_ID = {
    'Car': 1,
    'Van': 2,
    'Truck': 3,
    'Pedestrian': 4,
    'Person_sitting': 5,
    'Cyclist': 6, 
    'Tram': 7,
    'Misc': 8,
    'DontCare': 9,
    'Person': 4
}

# def get_class_ID(name):
#     if name in class_ID:
#         return class_ID[name]
#     else:
#         max_id = max(class_ID.values())
#         class_ID[name] = max_id + 1
#         print(f'added class `{name}` to dict with value {max_id+1}')
#         return class_ID[name]

def convert_kitti():
    tasks = glob(ori_img_path+'/*')
    tasks = [t.split('/')[-1] for t in tasks]
    print(f'Found images for task: {tasks}')

    for task in tasks:
        print(f'Processing task: {task}')
        os.makedirs(f'{target_txt_path}/{task}', exist_ok=True)
        os.makedirs(f'{target_img_path}/{task}', exist_ok=True)
        label_file = glob(f'{ori_label_path}/{task}.txt')[0]
        
        label_df = pd.read_csv(label_file, names=['frame', 'track_id', 'type', 'truncated', 
            'occluded', 'alpha', 'left', 'top', 'right', 'bottom', 'height', 'width', 'length'
                'x', 'y', 'z', 'r_y', 'score'], sep=' ')
        #ge img size
        rand_img = random.choice(glob(f'{ori_img_path}/{task}/*.png'))
        img = Image.open(rand_img)
        w, h = img.width, img.height
        label_df['x_center'] = (label_df.left + label_df.right)/2/w
        label_df['y_center'] = (label_df.top + label_df.bottom)/2/h
        label_df['w_p'] = (label_df.right - label_df.left)/w
        label_df['h_p'] = (label_df.bottom - label_df.top)/h
        label_df['class'] = label_df.type.apply(lambda x: class_ID[x])
        frames = set(label_df.frame.to_list())
        for f in tqdm(frames):
            txt_path = f'{target_txt_path}/{task}/{f:06}.txt'
            img_path = f'{target_img_path}/{task}/{f:06}.jpg'
            if not os.path.exists(img_path):
                img = Image.open(f'{ori_img_path}/{task}/{f:06}.png')
                assert w == img.width and h == img.height
                img.save(img_path)
            if not os.path.exists(txt_path):
                txt_df = label_df.query('frame == @f')[['class', 'track_id', 'x_center', 'y_center', 'w_p', 'h_p']]
                txt_df.to_csv(txt_path, header=False, index=False, sep=' ')


# validate generated annotation file
def render_img(img_path, txt):
    img = Image.open(img_path)
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img, 'RGBA')
    anno = pd.read_csv(txt, names=['class', 'track_id', 'x_center', 'y_center', 'w_p', 'h_p'], sep=' ')
    anno['x1'] = (anno.x_center - anno.w_p/2) * w
    anno['y1'] = (anno.y_center - anno.h_p/2) * h
    anno['x2'] = (anno.x_center + anno.w_p/2) * w
    anno['y2'] = (anno.y_center + anno.h_p/2) * h
    for i, row in anno.iterrows():
        draw.rectangle([(row.x1, row.y1), (row.x2, row.y2)])
    img.show(title='anno')


def show_rand_img():
    imgs = glob(target_img_path+'/**/*.jpg', recursive=True)
    img = random.choice(imgs)
    frame = img.split('/')[-1].split('.')[0]
    print(f'ploting task: {frame}')
    txt = img.replace('images', 'labels_with_ids').replace('.jpg', '.txt')
    render_img(img, txt)

def gen_train_val():
    tasks = glob(target_img_path+'/*')
    tasks = [t.split('/')[-1] for t in tasks]
    #split train val
    train_set = tasks[:int(len(tasks)*0.8)]
    val_set = tasks[int(len(tasks)*0.8):]

    with open(f'{data_root}/{data_root}.train', 'w') as f:
        for task in train_set:
            images = glob(f'{target_img_path}/{task}/*.jpg')
            for t in images:
                f.write(t+'\n')

    with open(f'{data_root}/{data_root}.val', 'w') as f:
        for task in val_set:
            images = glob(f'{target_img_path}/{task}/*.jpg')
            for t in images:
                f.write(t+'\n')


if __name__ == '__main__':
    # convert_kitti()
    # gen_train_val()
    show_rand_img()
