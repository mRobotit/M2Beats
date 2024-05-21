import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class mmpose:
    def __init__(self):
        det_config = "mmdet_config/yolox/yolox_s_8xb8-300e_coco.py"
        det_checkpoint = "checkpoint/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
        pose_config = "mmpose_config/body_2d_keypoint/yoloxpose/coco/yoloxpose_s_8xb32-300e_coco-640.py"
        pose_checkpoint = "checkpoint/yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth"

        #人物检测
        # det_config = "demo/mmdetection_cfg/yolox_tiny_8x8_300e_coco.py"
        # det_checkpoint = "checkpoint/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
        #resnet152 256*192精度不行且慢
        # pose_config = "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res152_coco_wholebody_256x192.py"
        # pose_checkpoint = "checkpoint/res152_coco_wholebody_256x192-5de8ae23_20201004.pth"
        #resnet50 256*192精度不行
        # pose_config = "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_256x192.py"
        # pose_checkpoint = "checkpoint/res50_coco_wholebody_256x192-9e37ed88_20201004.pth"
        #resnet50 384x288精度不行
        # pose_config = "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_384x288.py"
        # pose_checkpoint = "checkpoint/res50_coco_wholebody_384x288-ce11e294_20201004.pth"

        #手部检测
        # det_config = r"demo/mmdetection_cfg/jdxlite_yolov3.py"
        # det_checkpoint = "checkpoint/yolov3.pth"
        #手部关键点
        #mobelnetv2
        # pose_config = r"configs/hand/2d_kpt_sview_rgb_img/deeppose/onehand10k/res50_onehand10k_256x256.py"
        # pose_checkpoint = "checkpoint/deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth"


        device = 'cuda:0'
        assert has_mmdet, 'Please install mmdet to run the demo.'
        draw_heatmap = False

        # build detector
        self.detector = init_detector(det_config, det_checkpoint, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))

        # build visualizer
        self.pose_estimator.cfg.visualizer.radius = 3
        self.pose_estimator.cfg.visualizer.alpha = 0.8
        self.pose_estimator.cfg.visualizer.line_width = 1
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta, skeleton_style='mmpose')

    def get_keypoint(self, img):
        det_cat_id = 1
        bbox_thr = 0.3
        nms_thr = 0.3

        # predict bbox
        det_result = inference_detector(self.detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id, pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]

        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        return pose_results

    def draw(self, img, pose_results):
        draw_heatmap = False
        draw_bbox = True
        show_kpt_idx = False
        skeleton_style = "mmpose"
        show = False
        show_interval = 0
        kpt_thr = 0.3

        data_samples = merge_data_samples(pose_results)
        self.visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=draw_heatmap,
            draw_bbox=draw_bbox,
            show_kpt_idx=show_kpt_idx,
            skeleton_style=skeleton_style,
            show=show,
            wait_time=show_interval,
            kpt_thr=kpt_thr)
        return self.visualizer.get_image()
