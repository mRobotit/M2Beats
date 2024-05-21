import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from module.m2bnet import m2bnet

from visualize import beat_on_movie

def inference(keypoint_path, movie_name, path_saved_ckpt, fps, output_path):
    conf = 0.7
    
    print("load keypoint file {}".format(keypoint_path))

    # init
    net = m2bnet()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    if torch.cuda.is_available():
        net.cuda()
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt))
    else:
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt, map_location=torch.device('cpu')))

    fr = np.load(keypoint_path, allow_pickle=True)
    keypoints = fr['keypoints'][:1024] # 9999,34
    mask = fr['mask']

    keypoints = torch.tensor([keypoints], dtype=torch.float).cuda()
    with torch.no_grad():
        activation = net(keypoints)
    activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()

    length = np.where(mask==1)[0][-1]
    activation = activation[:length]

    # 可视化
    show_start = 0
    show_end = 1000
    activation_show = np.array(activation*100, int)
    activation_show = activation_show[show_start:show_end]
    x = np.arange(len(activation_show))
    plt.plot(x, activation_show, color='g')
    os.makedirs("./output/figs",exist_ok=True)
    plt.savefig('./output/figs/{}.png'.format(movie_name))

    # 聚合
    act_pos = np.array(gather_act(activation, conf, fps), int)
    print("act_pos: {}".format(act_pos))

    # # 视频可视化
    # if args.movie:
    #     beat_on_movie(args.movie, act_pos, fps)

    # 保存推断结果
    os.makedirs(output_path,exist_ok=True)
    inf_output_file = os.path.join(output_path, "{}.npz".format(movie_name))
    np.savez(inf_output_file, beat=act_pos)

def gather_act(activation, conf, fps):
    x = fps/6 # 聚簇距离 10帧,1/6秒
    K = fps/3 # 节拍最小间距

    right = 0
    gathering = False
    max_num = 0
    max_idx = 0

    act_pos = []

    for i in range(len(activation)):
        # 超过聚簇距离
        if gathering and i > right:
            act_pos.append(max_idx)
            gathering = False

        if activation[i] > conf:
            # 开始聚簇
            if not gathering:
                gathering = True
                max_num = activation[i]
                max_idx = i
            # 比原来那个大
            elif activation[i] > max_num:
                max_num = activation[i]
                max_idx = i
            # 刷新右侧极限
            if i + x < len(activation):
                right = i + x
            else:
                right = len(activation) - 1
    if gathering:
        act_pos.append(max_idx)

    return act_pos
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for training CMT")
    parser.add_argument('-i', '--input', help="The input keypoint path")
    parser.add_argument('-c', '--checkpoint', help="The model checkpoint path")
    parser.add_argument('-m', '--movie', help="The model checkpoint path")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', default=[0], help="Ids of gpu")
    parser.add_argument('--fps', type=int, default=60, help="fps of movie")
    args = parser.parse_args()
    if args.gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in list(range(torch.cuda.device_count()))])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])

    keypoint_path = args.input
    movie_name = os.path.basename(keypoint_path).split('.')[0]
    path_saved_ckpt = args.checkpoint
    fps = args.fps
    output_path = "./output/inf/"

    inference(keypoint_path, movie_name, path_saved_ckpt, fps, output_path)
