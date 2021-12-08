import os
import click
import json
from random import randint
from enum import Enum
from detectron2.structures.boxes import BoxMode
import numpy as np

class Tag(Enum):
    VAL = 0
    TRAIN = 1

def read_meta_json(path):
    '''
    reads meta.json which contains information about categories
    '''
    data = json.load(open(path))
    return {cl['title']: i for i, cl in enumerate(data['classes'])}


def match_img_to_ann(images_path, anns_path):
    '''
    splits into train (80%) and valid (20%) sets
    returns a list of (image_name, full_info, tag)
    '''
    all_images_names = [img for img in os.listdir(images_path)]
    all_annots_names = [ann for ann in os.listdir(anns_path)]
    matched_pairs = []
    for img in all_images_names:
        # take 90% to train set and the rest to validation set
        tag = Tag.TRAIN if randint(1, 10) in range(10) else Tag.VAL
        img_name = img.split('.')[0]
        try:
            annot = list(filter(lambda ann: '.jpg.json' in ann and img_name == ann.split('.')[0], all_annots_names))[0]
            all_annots_names.remove(annot)
            matched_pairs.append((f'{images_path}/{img}', f'{anns_path}/{annot}', tag))
        except:
            continue
    return matched_pairs


def polygon_to_bbox(polygon):
    '''
    converts a polygon into a bounding box
    returns in format XYXYABS
    '''
    xs, ys = zip(*polygon)
    return (min(xs), max(ys), max(xs), min(ys))


def retrieve_info(image_path, json_path, categories):
    '''
    retrieves informatation about image from image_path by using json_path
    which should correspond to the image
    '''
    assert '.json' in json_path
    data = json.load(open(json_path))
    info = {
        'height': data['size']['height'],
        'width': data['size']['width'],
        'image_id': image_path.split('/')[-1].split('.')[0],
        'file_name': image_path,
        'annotations': [
            {
                'bbox_mode': BoxMode.XYXY_ABS,
                'bbox': polygon_to_bbox(obj['points']['exterior']),
                # 'segmentation': obj['points']['exterior'],
                'segmentation': [list(map(int, list(np.array(obj['points']['exterior']).reshape(-1))))],
                'category_id': categories[obj['classTitle']]
            } for obj in data['objects']
        ]
    }
    return info


@click.command()
@click.option('--images-path', default='recycling-codes-dataset/photos/img', prompt='Path to the images', type=str)
@click.option('--annotations-path', default='recycling-codes-dataset/photos/ann', prompt='Path to the annotations', type=str)
@click.option('--meta-path', default='recycling-codes-dataset/meta.json', prompt='Path to meta.json file', type=str)
@click.option('--save-train-to', default='train.json', prompt='Save train dataset to:', type=str)
@click.option('--save-valid-to', default='valid.json', prompt='Save valid dataset to:', type=str)
def main(images_path: str, annotations_path: str, meta_path: str, save_train_to: str, save_valid_to: str):
    imgs_to_annot = match_img_to_ann(images_path, annotations_path)
    categories = read_meta_json(meta_path)
    train_infos, valid_infos = [], []
    for img, ann, tag in imgs_to_annot:
        info = retrieve_info(img, ann, categories)
        if tag == Tag.TRAIN:
            train_infos.append(info)
        else:
            valid_infos.append(info)

    train_json_objects = json.dumps(train_infos, indent=4)
    valid_json_objects = json.dumps(valid_infos, indent=4)
    with open(save_train_to, 'w') as fjs:
        fjs.write(train_json_objects)
    with open(save_valid_to, 'w') as fjs:
        fjs.write(valid_json_objects)
    print(f'Saved {len(train_infos)} to train.json and {len(valid_infos)} to valid.json!')

if __name__ == '__main__':
    main()
