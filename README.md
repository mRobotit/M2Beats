# M2Beats
Official code for IJCAI 2024 paper: M2Beats: When Motion Meets Beats in Short-form Videos

## 环境安装步骤

### 推荐环境 

Linux
conda
python 3.9

### 1.在pytorch官网选择*适合自己*的pytorch<2的版本

https://pytorch.org/get-started/locally/

参考安装命令：

`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

### 2.安装mmpose全家桶

`pip install -U openmim`
`mim install "mmcv==2.1.0"`
`mim install mmdet`
`mim install mmpose`

### 3.其他组件

`moviepy`

`mir_eval`

## 下载checkpoint和视频

checkpoint链接：

视频链接：

路径: `M2Beats/checkpoint/`

## wild video测试

修改demo.py中12-13行的movie_path和music_path为自己的视频和音乐

`python demo.py`

## 训练

### 数据集制作

`python make_dataset.py`

### 训练模型

`python train.py`

### 评估模型精度

python test.py