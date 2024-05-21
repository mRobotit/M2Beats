import os
import time
import cv2
from tqdm import tqdm
import numpy as np
from utils.class_mmpose import mmpose

mm = mmpose()


class Pose:
    def __init__(self, frame):
        self.frame = frame
        pose_result = mm.get_keypoint(frame)
        # vis_img = mm.draw(frame, self.pose_result)
        # cv2.imwrite(r"./JDX/{}.jpg".format(idx), vis_img)
        if len(pose_result) == 0:
            self.isHuman = False
            self.keypoints = []
        else:
            self.isHuman = True
            key = pose_result[0].pred_instances.keypoints[0]
            conf = pose_result[0].pred_instances.keypoint_scores[0]
            conf = np.expand_dims(conf, axis=1)
            self.keypoints = np.concatenate((key, conf),axis=1)
            assert self.keypoints.size == 51 # 3 * 17
        self.t = time.time()

def movie_keypoints(video_path, output_path):
    movie_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    assert frame < 1024, "视频应小于1024帧，提醒作者修改模型"

    keypoint_list = np.zeros((1024,17,3))
    mask = np.zeros((1024))
    pre_frame = None
    for i in tqdm(range(frame)):
        ret, frame = cap.read()
        if not ret:
            break
        pose = Pose(frame)
        if pose.isHuman is True:
            key = np.array(pose.keypoints)
            keypoint_list[i] = key
            mask[i] = 1
    output_file = os.path.join(output_path, movie_name + '.npz')
    np.savez(output_file, keypoints=keypoint_list, mask=mask)

        
if __name__ == "__main__":
    video_path = r"/data/jdx/code/mycode/movie/youying.mp4"
    print("video_path:{}".format(video_path))

    output_path = r"/data/jdx/code/mycode/output_wild/keypoints"

    movie_keypoints(video_path, output_path)