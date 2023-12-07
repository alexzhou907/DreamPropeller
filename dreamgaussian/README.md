# DreamGaussian

We directly provide code for speeding up DreamGaussian in the original codecase.

## Installation

Note the installation process is different from the original repo as we have a customized `./diff_gaussian_rasterization` to work on multiple GPUs.

```bash
pip install -r requirements.txt

pip install ./diff_gaussian_rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast if you have not installed it
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit for visualization
pip install git+https://github.com/ashawkey/kiuikit
```


## Usage

Set `FILE` to either `main_baseline.py` for `main_speedup.py`. 

Note that we only implemented speedup for the `main` loop for demonstration purposes as `main2.py` is a refinement process that takes a considerably shorter time. However, one can use similar logic to speed up `main2.py` as well. 

For quick tests,

Text-to-3D:
```
python $FILE --config configs/text.yaml prompt="a photo of an icecream"  batch_size=16 save_path=icecream

python main2.py --config configs/text.yaml prompt="a photo of an icecream" batch_size=1 save_path=icecream

## for visualization
python -m kiui.render logs/icecream.obj --save_video logs/icecream.mp4 --wogui
```

Image-to-3D:
```
### preprocess
# background removal and recenter, save rgba at 256x256
python process.py data/name.jpg

# save at a larger resolution
python process.py data/name.jpg --size 512

# process all jpg images under a dir
python process.py data

python $FILE --config configs/image.yaml input=data/sword_rgba.png batch_size=16 save_path=sword

python main2.py --config configs/image.yaml input=data/sword_rgba.png batch_size=1 save_path=sword

## for visualization
python -m kiui.render logs/sword.obj --save_video logs/sword.mp4 --wogui
```

Please check `./configs/text.yaml` and `./configs/image.yaml` for more options.
