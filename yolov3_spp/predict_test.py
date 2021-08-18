import os
import json
import random
import time

import torch
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from build_utils import img_utils, utils
from yolov3_spp.build_utils import torch_utils
from models import Darknet
from draw_box_utils import draw_box

nowWeight = open("weightPath.txt", mode='r')
nowWeightPath = nowWeight.read()
random.seed(time.time())

def main():
    matplotlib.use('TkAgg')
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = nowWeightPath  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    video_path = "/Users/softwind/Downloads/QQ/IMG_1209.MP4"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(video_path), "image file {} dose not exist.".format(video_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()

        dirName = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

        dataPath = open("dataPath.txt", mode='w')
        dataPath.write(f"{dirName}")
        dataPath.close()

        os.mkdir(f"{dirName}")
        os.chdir(f"{dirName}")
        os.makedirs("./train/images")
        os.makedirs("./train/labels")
        os.makedirs("./val/images")
        os.makedirs("./val/labels")

        saveCount = 0
        while success:
            if saveCount % 30 != 0:
                success, frame = capture.read()
                saveCount += 1
                continue

            saveCount += 1
            img_o = frame  # BGR

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension

            pred = model(img)[0]  # only get inference result

            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]

            if pred is None:
                print("No target detected.")
                success, frame = capture.read()
                continue

            # process detections
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()

            bboxes = pred[:, :4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(int)
            # scores = pred[:, 4].detach().cpu().numpy()
            # img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)  //预测


            OutputName = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
            fileOutputName = f"{OutputName}.txt"
            imgOutputName = f"{OutputName}.jpg"
            if random.randint(0, 9) < 7:
                os.chdir("train")
            else:
                os.chdir("val")

            cv2.imwrite(f"./images/{imgOutputName}", frame)
            i = 0
            while i < len(bboxes):
                x1 = (bboxes[i][0] + bboxes[i][2]) / (capture.get(3) * 2)
                y1 = (bboxes[i][1] + bboxes[i][3]) / (capture.get(4) * 2)
                x2 = bboxes[i][2] / capture.get(3)
                y2 = bboxes[i][3] / capture.get(4)
                line = f"{classes[i]} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}"
                if i == 0:
                    fileOutput = open(f"./labels/{fileOutputName}", mode='w')
                else:
                    fileOutput = open(f"./labels/{fileOutputName}", mode='a')
                    fileOutput.write('\n')

                fileOutput.write(line)
                fileOutput.close()
                i = i + 1

            os.chdir("..")
            success, frame = capture.read()

    print("Finished!")


if __name__ == "__main__":
    print(nowWeight)
    main()
