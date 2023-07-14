from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import json
# 使用测试函数进行目标检测

def statistical_category(result):
    
    labels = result.pred_instances.labels.numpy()  # .tolist()
    # 把标签和名字对应起来
    res=labels+1
    unique_values, counts = np.unique(res, return_counts=True)

    # 创建Echarts配置选项
    options = {
        "title": {"text": "细胞种类及个数柱状图","x":"center"},
    "xAxis": {"type": "category", "data": ["1", "2", "3", "4", "5"]},
    "yAxis": {"type": "value"},
    "series": [{"data": counts.tolist(), "type": "bar"}],
    }

    return options


def predict(params,iou):
    config_file = params['config_path']
    checkpoint_file = params['epochs']
    img_path = params['img_path']
    img_name = params['filename']
    cfg_options = {
        'model.test_cfg.rcnn.score_thr': iou
    }
    model = init_detector(config_file, checkpoint_file, device='cpu',cfg_options=cfg_options)
    result = inference_detector(model, img_path)
    img = mmcv.imread(img_path,
                      channel_order='rgb')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    res = './result/'+img_name
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file=res
    )
    return res,statistical_category(result)



def test(config_file, checkpoint_file, img_path, img_name):
    cfg_options = {
        'model.test_cfg.rcnn.score_thr': 0.8
    }
    model = init_detector(config_file, checkpoint_file, device='cpu',cfg_options=cfg_options)
    result = inference_detector(model, img_path)
    print(result)
    # print(result.pred_instances.scores)
    # print(result.pred_instances.bboxes)
    # print(result.pred_instances.labels)
    img = mmcv.imread(img_path,channel_order='rgb')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0
    )
    visualizer.show()
    
def main():
    config_file = 'config_model\\cascade.py'
    checkpoint_file = 'work_dirs\\cascade_epoch_12.pth'
    img_path = 'uploaded_files\\Uterus_2623.jpg'
    img_name = 'test_result.png'
    test(config_file, checkpoint_file, img_path, img_name)

if __name__ == '__main__':
    main()
