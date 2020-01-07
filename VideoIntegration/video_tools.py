#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from moviepy.editor import *
from PIL import Image

import os
import math
import numpy as np
import random
import cv2

# get image from video
# offset_t: skip offset_t time to start get images
# interval = get image every time in seconds
def get_image_from_video(video, offset_t = 0, interval = 300):
    
    # get the end time
    for time_index in range(int(video.duration - offset_t), int(video.duration), interval):
    
        #plt.subplot(10, 1, imageIndex)
        #print(frameIndex)
        f = video.get_frame(time_index)

        img = Image.fromarray(f, 'RGB')
        img.save('../data/%s_%d.jpg' % (vedio.file_name, time_index))
        print('/%s_%d.jpg.jpg saved' % (vedio.file_name, time_index))

    return end_time  

