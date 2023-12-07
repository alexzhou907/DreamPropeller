# DreamPropeller: Supercharge Text-to-3D Generation with Parallel Sampling

Official Implementation of [DreamPropeller: Supercharge Text-to-3D Generation with Parallel Sampling](https://arxiv.org/abs/2311.17082)

<p align="center">
  <img src="assets/teaser.gif" width="90%" loop=infinite/>
</p>


<div align="center">
<a href="https://arxiv.org/abs/2311.17082">[Paper]</a>
<a href="https://alexzhou907.github.io/dreampropeller_page/">[Project]</a>
</div>

----

This codebase is based on [threestudio](https://github.com/threestudio-project/threestudio) with minor changes to some files for parallel processing.

## Installation

- Install `Python >= 3.11`.
- Create a virtual environment:

```sh
conda create -n dreampropeller python=3.11
```


- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

- Install `PyTorch == 2.1.0`. We have tested on `torch2.1.0+cu121`.

```sh
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```


- (Optional, Recommended) The best-performing models in threestudio use the newly-released T2I model [DeepFloyd IF](https://github.com/deep-floyd/IF), which currently requires signing a license agreement. If you would like to use these models, you need to [accept the license on the model card of DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login into the Hugging Face hub in the terminal by `huggingface-cli login`.


## Usage


We can compare the runtime for baseline and DreamPropeller by using `launch_baseline.py` and `launch_speedup.py` respectively. For DreamPropeller, one can tune `speedup` parameters in `.yaml` config files. The preset values in the files are the default values. We note their meaning below:
- `speedup.threshold` denotes the threshold for fixed-point error. Larger threshold gives greater speedup but lower generation quality.
- `speedup.ema_decay` denotes the EMA rate of updating the threshold. This parameter makes it robust against variation in speedup for different prompts.
- `speedup.P` denotes the window size, typically set to `num_gpu-1`. e.g. 8 GPU cluster uses window size of 7.
- `speedup.adaptivity_type` denotes the error aggregation function used to update `speedup.ema_decay`. We only support `median` and `mean` and we use `median` for all but ProlificDreamer. We find using `median` makes quality more stable for tested frameworks but gives slower speedup than `mean`.

Other visualization options:
- `trainer.visualize_progress` sets whether to visualize the generation progress.
- `trainer.display_time` sets whether to display current runtime for each displayed image.
- `train_config.val_check_interval` denotes the wallclock time in seconds (instead of iteration number) after which the script will save a visualization.

We can quickly test each of the frameworks presented in the paper below. Our method is tested on an 8-GPU A100-80G PCIe cluster.

You can freely switch between baseline and DreamPropeller by simply setting `FILE` to `launch_baseline.py` or `launch_speedup.py`.

### **DreamFusion**

```
python $FILE --config configs/dreamfusion-if.yaml --train system.prompt_processor.prompt="an ice cream sundae" data.batch_size=16 trainer.visualize_progress=true
```

### **Magic3D**

```
python $FILE  --config configs/magic3d-coarse-if.yaml --train system.prompt_processor.prompt="a beautiful dress made out of fruit, on a mannequin. Studio lighting, high quality, high resolution" data.batch_size=16 trainer.visualize_progress=true

python $FILE --config configs/magic3d-refine-sd.yaml --train system.prompt_processor.prompt="a beautiful dress made out of fruit, on a mannequin. Studio lighting, high quality, high resolution" data.batch_size=16 trainer.visualize_progress=true system.geometry_convert_from=path/to/coarse/stage/trial/dir/ckpts/last.ckpt
```

### **TextMesh**

```
python $FILE --config configs/textmesh-if.yaml --train system.prompt_processor.prompt="an old vintage car" data.batch_size=16 trainer.visualize_progress=true
```


### **ProlificDreamer**

```
python $FILE --config configs/prolificdreamer.yaml --train system.prompt_processor.prompt="a detailed Victorian era house" data.batch_size=[8,2] trainer.visualize_progress=true

python $FILE --config configs/prolificdreamer-geometry.yaml --train system.prompt_processor.prompt="a detailed Victorian era house" data.batch_size=8 trainer.visualize_progress=true system.geometry_convert_from=path/to/coarse/stage/trial/dir/ckpts/last.ckpt

python $FILE --config configs/prolificdreamer-texture.yaml --train system.prompt_processor.prompt="a detailed Victorian era house"  data.batch_size=1 trainer.visualize_progress=true system.geometry_convert_from=path/to/geometry/stage/trial/dir/ckpts/last.ckpt

```

### **DreamGaussian**

Since DreamGaussian is a separate standalone repo, we have copied it in the `dreamgaussian/` subfolder with our dreampropeller implementation specifically tailored to this repo. Please find specific instructions to run DreamGaussian inside.


### **Zero 1-to-3**

Our method can also be applied to Image-to-3D using score distillation. Luckily Zero 1-to-3 is implemented in the original threestudio code, so we can directly plug in our method.

Download pretrained Zero123XL weights into `load/zero123`:

```sh
cd load/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```

To run,

```
python $FILE --config configs/zero123.yaml --train data.image_path=./load/images/grootplant_rgba.png data.random_camera.batch_size=[16,16,10] trainer.visualize_progress=true
```


## Reference

```
@article{zhou2023dreampropeller,
  title={DreamPropeller: Supercharge Text-to-3D Generation with Parallel Sampling},
  author={Zhou, Linqi and Shih, Andy and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2311.17082},
  year={2023}
}
```
