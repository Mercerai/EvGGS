# EvGGS: A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting
[A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting](https://arxiv.org/abs/2405.14959v1) 

Jiaxu Wang, Junhao He, Ziyi Zhang, Mingyuan Sun, Jingkai Sun, Renjing Xu*

<p align="center">
<img src="./Figures/network.png" width="1000"><br>
Fig  1. The main pipeline overview of the proposed EvGGS framework.
</p>

# Create environment
```
conda env create --file environment.yml
conda activate evggs
```
Then, compile the diff-gaussian-rasterization in 3DGS repository:
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..
```
# Download models
Download the pretrained models from [OneDrive](https://hkustgz-my.sharepoint.com/:u:/g/personal/jwang457_connect_hkust-gz_edu_cn/ESAMKY3oHDRBr2-zeNb3L8IBKnFGiJCAgyRv3HBs6esFaQ?e=O7bili) that are placed at ```\pretrain_ckpt```. This directory includes two warmup ckpts and a pretrained ckpts on the synthetic dataset.

# Running the code

## Download dataset

- Ev3D-S

    A large-scale synthetic Event-based dataset with varying textures and materials accompanied by well-calibrated frames, depth, and groundtruths. 

    You can download the dataset from [OneDrive](https://hkustgz-my.sharepoint.com/:u:/g/personal/jwang457_connect_hkust-gz_edu_cn/EYszUyxQnzRMkC0u5GxDOvEB_NhmBaVe2vBnpMH2ctSWxA?e=kJDwRz) and unzip it. A 50 GB of storage space is necessary.


- EV3D-R

    A large-scale realistic Event-based 3D dataset containing various objects captured by a real event camera DVXplore.

    Due to some licensing reasons, we currently need your private application to use this dataset, but this will be addressed soon.
  
## Training

```
python train_gs.py
```

## Evaluation

```
python eval_gs.py
```

In ```configs\Ev3D_pretrain```, several primary settings are defined such as experimental name, customized dataset path, please check. 

# Citation

please cite our work if you use this dataset.

```
@misc{wang2024evggscollaborativelearningframework,
      title={EvGGS: A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting}, 
      author={Jiaxu Wang and Junhao He and Ziyi Zhang and Mingyuan Sun and Jingkai Sun and Renjing Xu},
      year={2024},
      eprint={2405.14959},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.14959}, 
}
```

# Reference

EventNeRF: [https://github.com/r00tman/EventNeRF?tab=readme-ov-file](https://github.com/r00tman/EventNeRF?tab=readme-ov-file).
3D Gaussian Splatting: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).
GPS-GS: [https://github.com/aipixel/GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian)
PAEvD3d: [https://github.com/Mercerai/PAEv3d](https://github.com/Mercerai/PAEv3d)
