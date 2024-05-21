import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd

from module.m2bnet import m2bnet
from scoreboard import precision_recall

def main():
    # note_file_dir = "/data/jdx/code/mycode/train_data"
    # movies = os.listdir(note_file_dir)

    test_list_file = "DATASET_ASSET/test_data.csv"
    movies = pd.read_csv(test_list_file)
    for conf in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]:
        precision = 0
        recall = 0
        # for movie in movies:
        for idx in tqdm(range(len(movies))):
            movie_name = movies.iloc[idx, 0]
            p, r = inference(movie_name, args.fps, conf)
            precision += p
            recall += r
        precision /= len(movies)
        recall /= len(movies)

        print("conf=%f,precision=%f\trecall=%f"%(conf,precision,recall))



def inference(movie_name, fps, conf):
    note_file_dir = "DATASET_ASSET/train_data"
    input_file = os.path.join(note_file_dir, movie_name + '.npz')
    # print("load movie file {}".format(input_file))

    fr = np.load(input_file, allow_pickle=True)
    keypoints = fr['keypoints'][:1024] # 9999,34
    beat = fr['beat'] # 5000,1   beat
    mask = fr['mask']

    keypoints = torch.tensor([keypoints], dtype=torch.float).cuda()
    t0 = time.time()
    with torch.no_grad():
        activation = net(keypoints)
    t1 = time.time()
    activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
    t2 = time.time()
    

    length = np.where(mask==1)[0][-1]
    beat = beat[:length]
    activation = activation[:length]

    beat_pos = np.where(beat==1)[0]
    t3 = time.time()
    act_pos = np.array(gather_act(activation, conf=conf), int)
    t4 = time.time()

    # # print("beat_pos: {}".format(beat_pos))
    # # print("act_pos: {}".format(act_pos))

    # show_start = 0
    # show_end = 1000
    # activation_show = np.array(activation*100, int)
    # activation_show = activation_show[show_start:show_end]
    # t5 = time.time()


    # beat_show = np.array(200-beat*100, int)
    # beat_show = beat_show[show_start:show_end]
    # x = np.arange(len(beat_show))
    # plt.plot(x, beat_show, color='r')


    # x = np.arange(len(activation_show))
    # plt.plot(x, activation_show, color='g')
    # t6 = time.time()


    # os.makedirs("./output/figs",exist_ok=True)
    # plt.savefig('./output/figs/{}.png'.format(movie_name))
    # plt.cla()
    # t7 = time.time()

    # # human_list = [[0,9],[0,43],[1,12],[1,43],[2,16],[2,48],[3,20],[3,54],[4,28],[5,1],[5,34],[6,6],[6,40],[7,13],[7,44],[8,15]]
    # # human_pos = ret_60_100(human_list)
    
    # # 保存推断结果
    # os.makedirs("./output/inf/",exist_ok=True)
    # inf_output_file = "./output/inf/{}.npz".format(movie_name)
    # np.savez(inf_output_file, beat=act_pos)
    # t8 = time.time()

    # print("01:%f\t02:%f\t34:%f\t56:%f\t67:%f\t78:%f" % (t1-t0,t2-t0,t4-t3,t6-t5,t7-t6,t8-t7), end='\r')
    
    precision, recall = precision_recall(beat_pos, act_pos)
    return precision, recall

def gather_act(activation, conf):
    x = 10 # 聚簇距离 10帧,1/6秒
    K = 20 # 节拍最小间距

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
                if i + x < len(activation):
                    right = i + x
                else:
                    right = len(activation) - 1
                gathering = True
                max_num = activation[i]
                max_idx = i
            # 比原来那个大
            elif activation[i] > max_num:
                max_num = activation[i]
                max_idx = i
    if gathering:
        act_pos.append(max_idx)

    return act_pos
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args for training CMT")
    # parser.add_argument('-i', '--input', help="The input name")
    parser.add_argument('-c', '--checkpoint', help="The model checkpoint path")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', default=[0], help="Ids of gpu")
    parser.add_argument('--fps', type=int, default=60, help="fps of movie")
    args = parser.parse_args()

    if args.gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in list(range(torch.cuda.device_count()))])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])
    
    path_saved_ckpt = args.checkpoint
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

    main()
