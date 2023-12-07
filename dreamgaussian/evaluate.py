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
from random import randrange
import numpy as np
def r_precision(candidates, num_chunks=10):
    predictions = []
    for cand in candidates:
        if cand["prediction"] == cand['gt']:
            predictions.append(1)
        else:
            predictions.append(0)

    num_preds = len(predictions)
    chunk_size = int(num_preds / num_chunks)

    predictions = np.array(predictions)
    np.random.shuffle(predictions)

    chunks = np.zeros(num_chunks)
    for i in range(num_chunks):
        chunks[i] = np.average(predictions[i * chunk_size : (i + 1) * chunk_size])

    return np.average(chunks), np.std(chunks)

def get_r_precision(args):


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # model creation
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = model.cuda()
    model.eval()

    
    with open('../threestudio/load/prompt_subset.json') as json_data:
        prompts = json.load(json_data)
        json_data.close()
    
    # run prediction
    clip_result = []
    for cap_id, p in tqdm(enumerate(prompts)):
        save = p.replace(' ', '_')
        img_paths = glob.glob(os.path.join(args.dir_path , f'{save}*/save/*-test/*.png'))
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")

            img = img.resize((512,512)).crop((0, 0, 512, 512))
            
            inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to('cuda')

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[0, 0].item()

            new_entry = ( cap_id, img_path, clip_prediction)
            clip_result.append(new_entry)

    with open(os.path.join(args.dir_path, 'clip_r_precision.json'), 'wb') as f:
        pickle.dump(clip_result, f)

    candidates = []
    for entry in clip_result:
        cap_id, gen_img_path, r_precision_prediction = entry
        candidates.append({
            "prediction": r_precision_prediction,
            "img_path": gen_img_path,
            'gt': cap_id
        })
    R_mean, R_std = r_precision(candidates)
    
    print("R Precision score: ")
    print(f"\t {R_mean * 100:.2f} +- {R_std * 100:.2f}")


def get_fid(args):
    from cleanfid import fid


    
    with open('../threestudio/load/prompt_subset.json') as json_data:
        prompts = json.load(json_data)
        json_data.close()

    out_dir = os.path.join(args.dir_path, 'all_renders')
    os.makedirs(out_dir, exist_ok=True)

    cnt = 0
    for cap_id, p in tqdm(enumerate(prompts)):
        save = p.replace(' ', '_')
        img_paths = glob.glob(os.path.join(args.dir_path , f'{save}*/save/*-test/*.png'))
        for f in img_paths:
            img = Image.open(f).convert("RGB")

            img = img.resize((512,512)).crop((0, 0, 512, 512))
            for _ in range(20):
                matrix = 300
                x, y = img.size
                x1 = randrange(0, x - matrix)
                y1 = randrange(0, y - matrix)
                img_ = img.crop((x1, y1, x1 + matrix, y1 + matrix)).resize((512,512))
                img_.save(os.path.join(out_dir, str(cnt)+'.png'))

                cnt += 1

    score = fid.compute_fid(out_dir, args.imagenet_path, mode="clean", model_name="clip_vit_b_32")
    print('FID:', score)
    # with open(os.path.join(args.dir_path, 'fid_clip.json'), 'wb') as f:
    #     pickle.dump(score, f)

def get_runtime(args):

    import datetime

    with open(f'{args.dir_path}/metrics.json') as json_data:
        metrics = json.load(json_data)
        json_data.close()

    runtime = 0
    for data in metrics:
        # t = datetime.datetime.strptime(data['elasped'].split('.')[0],"%H:%M:%S")
        # delta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()
        runtime += float(data['elasped'])

    runtime = runtime / len(metrics)
    print('Hours:', str(datetime.timedelta(seconds=runtime)))
    print('Seconds:', runtime)

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dir_path", default="", type=str,)
    parser.add_argument("--out_path", default="", type=str,)
    parser.add_argument("--imagenet_path", default='', type=str)
    parser.add_argument("--metric", default="r-precision", type=str,)
    args = parser.parse_args()

    if args.metric == 'r-precision':
        get_r_precision(args)
    elif args.metric == 'fid':
        get_fid(args)
    elif args.metric == 'runtime':
        get_runtime(args)
