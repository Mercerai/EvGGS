# EvGGS: A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting
[A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting](https://arxiv.org/abs/2405.14959v1) 

Jiaxu Wang, Junhao He, Ziyi Zhang, Mingyuan Sun, Jingkai Sun, Renjing Xu*

<p align="center">
<img src="./Figures/network.png" width="1000"><br>
Fig  1. The overview of EvGGS.
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
Download the pretrained models and be expected at ```\pretrain_ckpt```


# Training and evaluation
```
python <path-to-your-code>/train_gs.py
python <path-to-your-code>/eval_gs.py
```

# Running on the synthetic and realistic datasets

## Download dataset

- Ev3D-S

    A large-scale synthetic Event-based dataset with varying textures and materials accompanied by well-calibrated frames, depth, and groundtruths. 

    You can download the dataset from OneDrive and unzip it. A 50 GB of storage space is necessary for the training and evaluation.


- EV3D-R

    A large-scale realistic Event-based 3D dataset containing various objects captured by a real event camera DVXplore.

    You can download the dataset from [OneDrive](https://hkustgz-my.sharepoint.com/:u:/g/personal/junhaohe_hkust-gz_edu_cn/EY__SmcUSbdFs13sb2h8svYBXYOCDd0OVnSWV-WLfvFLmA?e=GkVjhd) and [Baidu Disk](https://pan.baidu.com/s/1EuR-l_b_g-j_Du6dOxtZEg?pwd=3ilt ), code:3ilt.
  
## Training

```
python <path-to-your-code>/train_gs.py
```

## Evaluation

```
python <path-to-your-code>/eval_gs.py
```
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
