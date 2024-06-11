import os
import glob
import shutil
import argparse

from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--AICUP_dir', type=str, default='', help='your AICUP train dataset path')
    parser.add_argument('--YOLOv7_dir', type=str, default='', help='converted dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.95, help='The ratio of the train set when splitting the train set and the validation set')

    opt = parser.parse_args()
    return opt


def aicup_to_yolo(args):
    train_dir = os.path.join(args.YOLOv7_dir, 'train')
    valid_dir = os.path.join(args.YOLOv7_dir, 'valid')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    
    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)
    
    total_files = sorted(
        os.listdir(
            os.path.join(args.AICUP_dir, 'images')
        )
    )
    
    total_count = len(total_files)
    train_count = int(total_count * args.train_ratio)
    
    train_files = total_files[:train_count]
    valid_files = total_files[train_count:]
    
    for src_path in tqdm(glob.glob(os.path.join(args.AICUP_dir, '*', '*', '*')), desc=f'copying data'):
        text = src_path.split(os.sep)
        timestamp = text[-2]
        camID_frameID = text[-1]
        
        train_or_valid = 'train' if timestamp in train_files else 'valid'
        
        if 'images' in text:
            dst_path = os.path.join(args.YOLOv7_dir, train_or_valid, 'images', timestamp + '_' + camID_frameID)
        elif 'labels' in text:
            dst_path = os.path.join(args.YOLOv7_dir, train_or_valid, 'labels', timestamp + '_' + camID_frameID)
        
        shutil.copy2(src_path, dst_path)
    
    return 0


def delete_track_id(labels_dir):
    for file_path in tqdm(glob.glob(os.path.join(labels_dir, '*.txt')), desc='delete_track_id'):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            text = line.split(' ')
            
            if len(text) > 5:
                new_lines.append(line.replace(' ' + text[-1], '\n'))

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

    return 0


if __name__ == '__main__':
    args = arg_parse()
    
    # debug
    # args.AICUP_dir = './train'
    # args.YOLOv7_dir = './yolo'
    # args.train_ratio = 0.8
    
    aicup_to_yolo(args)
    
    train_dir = os.path.join(args.YOLOv7_dir, 'train', 'labels')
    val_dir = os.path.join(args.YOLOv7_dir, 'valid', 'labels')
    delete_track_id(train_dir)
    delete_track_id(val_dir)