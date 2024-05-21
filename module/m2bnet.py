import numpy as np
import torch
import torch.cuda
from torch import nn

from module.st_gcn.st_gcn_aaai18 import ST_GCN_18

def slide_window_to_sequence(slide_window,window_step,window_size):
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

    sequence = torch.stack(sequence)

    return sequence

class m2bnet(nn.Module):
    def __init__(self, d_model=128, nlayers=4, nhead=8, d_hid=2048, is_training=True):
        super(m2bnet, self).__init__()
        n_token = 1 # 1 beat 2 beat downbeat
        self.d_model = d_model

        graph_cfg = {
            "layout": 'coco',
            "strategy": "spatial"
        }
        self.pose_net = ST_GCN_18(3, d_model*4, 10, graph_cfg, dropout=0.1)

        self.linear1 = nn.Linear(d_model,256)
        self.linear2 = nn.Linear(256,d_model)
        self.relu = nn.ReLU()
        # self.lstm = LSTM(input_size=d_model,output_size=d_model)

        # # Transformer encoder layer 
        # encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=d_hid, batch_first=True)# Batch first
        # # Transformer encoder
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.proj_beat = nn.Linear(d_model, 1)

        # no tcn
        # self.add = nn.Linear(512, d_model)
    
    def forward_output(self, h):
        y_beat = self.proj_beat(h)
        
        return y_beat

    def compute_loss(self, predict, origin_target, target, loss_mask):
        #两侧加权loss
        loss1 = self.loss_func(predict, origin_target)
        loss2 = self.loss_func(predict, target)

        loss = loss1 + loss2
        
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss
    
    def forward_pose_net(self, pose):
        N, C, T, V, M = pose.size()
        pose = self.pose_net(pose)  # output: (batch, d_model*4, frame/4)
        pose = pose.permute(0,2,1)  # (batch, frame/4, d_model*4)
        pose = pose.reshape(N, T, self.d_model)  # (batch, frame, d_model)

        # no tcn
        # N, C, T, V, M = pose.size()
        # pose = self.pose_net(pose)  # output: (batch, d_model*4, frame/4)
        # pose = pose.permute(0,2,1)  # (batch, frame/4, d_model*4)
        # pose = pose.reshape(N, T, 512)  # (batch, frame, d_model)
        # pose = self.add(pose)

        return pose
    
    
    def forward(self, x):
        # batch = 1
        #x: (batch, frame, dmodel), FloatTensor
        # batch, frame, V, C = x.shape
        # x = x.view((batch,frame,V*C))

        N, T, V, C = x.size()
        x = x.permute(0,3,1,2)
        x = x.view((N,C,T,V,1))
        x = self.forward_pose_net(x)

        h = self.linear1(x)
        h = self.relu(h)
        h = self.linear2(h)
        h = self.relu(h)

        # no fc
        # h = x

        y_beat = self.forward_output(h)

        return y_beat
    
    def inference(self, x):
        h = self.forward_hidden(x)
        y_beat= self.forward_output(h)

        y_beat = y_beat[:, ...].permute(0, 2, 1)

        return y_beat
    



if __name__ == '__main__':
    pass