from moviepy.editor import *
from PIL import Image
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import math
import numpy as np
import random
import cv2
import datetime
import detectron2
import csv

starttime = datetime.datetime.now()

# offset_t: skip offset_t time to predict
# predict_object: predict target object, by COCO dateset (0 = person)
# predict rate in time 
def get_start_time(video, offset_t = 0, predict_object = 0, rate = 1):

    fps = math.ceil(video.fps)
    f_index = 0
    for f in video.iter_frames():
    
        # every {secondEvery} seconds take a image
        div = divmod(f_index, fps * rate)
        if f_index > fps * offset_t and div[1] == 0:
    
            #plt.subplot(10, 1, imageIndex)
            fbgr = f[...,::-1]
            outputs = predictor(fbgr)
            cla = outputs["instances"].pred_classes
            #print('Read %d frame: %s' % (f_index, cla))
            
            # predicte start time. have person in the frame.
            if len(cla) != 0 and cla.min() == 0:
                
                # when get the detectron object then set start time and skip to end
                start_time = math.floor(f_index / fps) - 1
                
                #v = Visualizer(f, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
                #img = Image.fromarray(v.get_image(), 'RGB')
                #img.save('../data/target_start_ffmpeg_%d.jpg' % (start_time + 1))
                #print('target_start_ffmpeg_%d.jpg saved' % (start_time + 1))
                
                # skip to end 
                break
                
        f_index += 1
        
    return start_time

# predict video object start at ending where skip offset time.
# offset_t: skip offset_t time from ending to predict
# limit_t: max times to predict object
# predict_object: predict target object, by COCO dateset (0 = person)
# predict rate in time 
def get_end_time(video, offset_t = 0, limit_t = 5, rate = 1):
    
    end_time = int(video.duration - offset_t - limit_t)
    
    # get the end time
    for time_index in range(int(video.duration - offset_t), end_time, -1):
    
        # every 2 seconds take a image
        div = divmod(time_index, rate)
        if div[1] == 0:
    
            #plt.subplot(10, 1, imageIndex)
            #print(frameIndex)
            f = video.get_frame(time_index)
            fbgr = f[...,::-1]
            
            outputs = predictor(fbgr)
            cla = outputs["instances"].pred_classes
            #print('Read %d frame: %s' % (time_index * math.ceil(video.fps), cla))
            # outputs is not empty and have 0(penson) get
            if (len(cla) != 0 and cla.min() == 0):
                
                #v = Visualizer(f, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
                #img = Image.fromarray(v.get_image(), 'RGB')
                #img.save('../data/target_end_ffmpeg_%d.jpg' % (time_index))
                #print('target_end_ffmpeg_%d.jpg saved' % (time_index))
                
                end_time = time_index + 1
                break
                
    return end_time  


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = "../models/faster_rcnn_R_50_FPN_3x_model_final_280758.pkl"
predictor = DefaultPredictor(cfg)


'''
# get start and end time from csv file
cut_times = {}
with open('/mnt/hgfs/shared/data/cut_times.csv') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        cut_times[row[0]] = row[1:]
    #print(cut_times)

'''

L = []
path = "/mnt/hgfs/shared/data/"
# loop the file in folder
for root, dirs, files in os.walk(path):
    
    if (path != root):
        continue
        
    # 按文件名排序
    files.sort()
    # 遍历所有文件
    for file in files:
        # 如果后缀名为 .mp4
        if os.path.splitext(file)[1] == '.mp4':
            # 拼接成完整路径
            filePath = os.path.join(root, file)
            
            # clip the video
            v = VideoFileClip(filePath)
            v_s, v_e = get_start_time(v, 8), get_end_time(v, 1)
            #v_s = int(cut_times[file][0])
            #v_e = int(cut_times[file][1])
            print('video %s cut duration is: %d, %d' % (filePath, v_s, v_e))
            
            v = v.subclip(v_s, v_e).fx(vfx.fadein, duration=2).fx(vfx.fadeout, duration=2)
            L.append(v)

'''
# clip the video
video1 = VideoFileClip("/mnt/hgfs/shared/data/001.mp4")
v1_s, v1_e = get_start_time(video1, 8), get_end_time(video1, 5)
print('video 1 cut duration is: %d, %d' % (v1_s, v1_e))
video1 = video1.subclip(v1_s, v1_e).fx(vfx.fadein, duration=5).fx(vfx.fadeout, duration=5)

video2 = VideoFileClip("/mnt/hgfs/shared/data/051.mp4")
v2_s, v2_e = get_start_time(video2, 8), get_end_time(video2, 5)
print('video 2 cut duration is: %d, %d' % (v2_s, v2_e))
video2 = video2.subclip(v2_s, v2_e).fx(vfx.fadein, duration=5).fx(vfx.fadeout, duration=5)

if video2.w != video1.w:
    video2 = video2.resize(width=video1.w)
'''

print('start to concatenate video...')
# 拼接视频
final_clip = concatenate_videoclips(L)
# 生成目标视频文件
final_clip.write_videofile("/mnt/hgfs/shared/data/target_vr.mp4", bitrate="12000k", remove_temp=False, audio_codec="aac")
print('completed!')

endtime = datetime.datetime.now()
print ('operation time is %d' % (endtime - starttime).seconds)



'''
# read frame by cv2
cap = cv2.VideoCapture("/mnt/hgfs/shared/data/001-czechvrfetish-3d-1920x960-60fps-smartphone_hq-trailer-1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frameIndex = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, f = cap.read()
    div = divmod(frameIndex, fps * 2)
    if ret == True and div[1] == 0:
        
        imageIndex = div[0] + 1
        outputs = predictor(f)
        cla = outputs["instances"].pred_classes
        print('Read %d frame: %s ' % (frameIndex, cla))
        
        # outputs is not empty and have 0(penson) get
        if (len(cla) != 0 and cla.min() == 0):
            
            v = Visualizer(f, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            cv2.imwrite('../data/target_cv2_%d.jpg' % (imageIndex), v.get_image())
            print('target_cv2_%d.jpg saved' % (imageIndex))
            
            break
    frameIndex += 1

# When everything done, release the capture
cap.release()
'''  
        
'''
# read frame by ffmpeg(get frame method)
video = VideoFileClip("/mnt/hgfs/data/shared/001-czechvrfetish-3d-1920x960-60fps-smartphone_hq-trailer-1.mp4")

#frameIndex = 0
fps = math.ceil(video.fps)

for timeIndex in range(0, int(video.duration)):

    # every 2 seconds take a image
    div = divmod(timeIndex, 2)
    if div[1] == 0:

        imageIndex = div[0] + 1
        #plt.subplot(10, 1, imageIndex)
        #print(frameIndex)
        f = video.get_frame(timeIndex)
        fbgr = f[...,::-1]
        
        outputs = predictor(fbgr)
        cla = outputs["instances"].pred_classes
        print('Read %d frame: %s' % (timeIndex * fps, cla))
        # outputs is not empty and have 0(penson) get
        if (len(cla) != 0 and cla.min() == 0):
            
            v = Visualizer(f, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
            img = Image.fromarray(v.get_image(), 'RGB')
            img.save('../data/target_ffmpeg_%d.jpg' % (imageIndex))
            print('target_ffmpeg_%d.jpg saved' % (imageIndex))
            
            break
        
    #frameIndex += 1
'''