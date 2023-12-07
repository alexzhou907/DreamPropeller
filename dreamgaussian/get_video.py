import os
import pickle
import torch

from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel
import glob
import json
import shutil
import numpy as np
import skvideo
skvideo.setFFmpegPath('/home/alexzhou/ffmpeg/ffmpeg-6.1')
import skvideo.io
def get_image(args):

    
    # with open('load/prompt_subset.json') as json_data:
    #     prompts = json.load(json_data)
    #     json_data.close()

    # out_dir = os.path.join(args.dir_path, 'all_renders')
    # os.makedirs(out_dir, exist_ok=True)
    # if os.path.exists('tmp'):
    #     shutil.rmtree('tmp')
    # os.makedirs('tmp', exist_ok=True)
    
    cnt = 0 
    

    img_path_1 = sorted( list(glob.glob(os.path.join(args.path, f'progress/*.png'))), key=lambda x: int(x.split('/')[-1].split('.')[0][2:-2]) ) + sorted( list(glob.glob(os.path.join(args.path.replace('logs', 'logs2'), f'progress/*.png'))), key=lambda x: int(x.split('/')[-1].split('.')[0][2:-2]) )
    process_len = len(img_path_1)
    baseline_len = len(list(glob.glob(os.path.join(args.baseline, f'progress/*.png'))) + list(glob.glob(os.path.join(args.baseline.replace('logs', 'logs2'), f'progress/*.png'))))

    img_path_1 = img_path_1 + [img_path_1[-1]] * (baseline_len - process_len)

    img_path =img_path_1 + sorted( list(glob.glob(os.path.join(args.path.replace('logs', 'logs2'), f'save/0-test/*.png')) ),  key=lambda x: int(x.split('/')[-1].split('.')[0]))
    img_baseline_path = sorted( list(glob.glob(os.path.join(args.baseline, f'progress/*.png'))), key=lambda x: int(x.split('/')[-1].split('.')[0][2:-2]) ) + sorted( list(glob.glob(os.path.join(args.baseline.replace('logs', 'logs2'), f'progress/*.png'))), key=lambda x: int(x.split('/')[-1].split('.')[0][2:-2]) ) + sorted( list(glob.glob(os.path.join(args.baseline.replace('logs', 'logs2'), f'save/0-test/*.png')) ),  key=lambda x: int(x.split('/')[-1].split('.')[0]))
    
    
    imgs = []

    
    for f, bf in zip(img_path, img_baseline_path):
        # if 'normal' in f.split('/')[-1] or 'rgb' in f.split('/')[-1]:
        #     os.remove(f)
        #     continue
        img = Image.open(f).convert("RGB")
        width, height = img.size
        factor = 512 / height 
        rgb = img.resize((int(width * factor),int(height * factor))).crop((0, 0, 512, 512))
        
        img_baseline = Image.open(bf).convert("RGB")
        width, height = img_baseline.size
        factor = 512 / height 
        rgb_baseline = img_baseline.resize((int(width * factor),int(height * factor))).crop((0, 0, 512, 512))
        
        rgb =  np.asarray(rgb)
        rgb_baseline =  np.asarray(rgb_baseline)
        
        img = np.concatenate([rgb_baseline, rgb], axis=1)
        imgs.append(img)
        
        # rgb.save(f'tmp/img{cnt}.png')
        cnt += 1

    imgs = np.stack(imgs, axis=0)

    
    skvideo.io.vwrite(args.out, imgs, outputdict={
        '-b': '2000k',
        # '-c:a': 'aac'
        })
    
    # os.system(f"ffmpeg -r 1 -i tmp/img%01d.png -vcodec mpeg4 -y {args.out}")


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--path", default="", type=str,)
    parser.add_argument("--baseline", default="", type=str,)
    parser.add_argument("--out", default="", type=str,)
    
    
    args = parser.parse_args()

    args.path = 'logs/the_leaning_tower_of_Pisa,_aerial_view'
    args.baseline = 'logs_baseline/the_leaning_tower_of_Pisa,_aerial_view'
    args.out = 'dreamgaussian_tower.mp4'

    get_image(args)