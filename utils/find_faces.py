#coding=utf-8
import os

import numpy as np
from PIL import Image

import cv2
import face_recognition

def find_from_image():

    input_path = "/home/repositories/test-data/faces/face_image_1"
    result_path = os.path.join("/home/repositories/test-data/", "faces/face_result_1")
    
    files= os.listdir(input_path)
    # 按文件名排序
    files.sort()
    
    # 遍历所有文件
    file_index = 0
    face_index = 0
    for file in files:

        # skip sub dirs
        if os.path.isdir(file):
            continue

        # 如果后缀名为.jpg
        ext = os.path.splitext(file)[1].lower()
        if ext in [".jpg", ".png"]:
            # 拼接成完整路径
            filePath = os.path.join(input_path, file)

        # load image file
        org_img = Image.open(filePath)
        w, h = org_img.size

        # 缩小图片来寻找人脸，更大的数值寻找更小的人脸。
        scale = 400 / w
        sw, sh = int(scale * w), int(scale * h)
        small_img = org_img.resize((sw, sh), Image.ANTIALIAS).convert('RGB')

        face_locations = face_recognition.face_locations(np.array(small_img), number_of_times_to_upsample=1)
        # Loop through each face found in the unknown image
        for top, right, bottom, left in face_locations:
            
            # get expand pix 
            inc_pix = (right - left) * 0.1

            # increate located face size 
            top = int((top - inc_pix) / scale)
            if top < 0: top = 0
            left = int((left - inc_pix) / scale)
            if left < 0: left = 0

            bottom = int((bottom + inc_pix) / scale)
            if bottom > h: bottom = h
            right = int((right + inc_pix) / scale)
            if right > w: right = w

            # access the actual face
            face_image = org_img.crop((left, top, right, bottom))

            # You can also save a copy of the new image to disk if you want by uncommenting this line
            save_path = os.path.join(result_path, ("%s_%s.jpg" % (os.path.splitext(file)[0], face_index)))
            face_image.save(save_path)
            face_index += 1

        file_index += 1
        print("\r %d/%d files completed. %d faces saved." % (file_index, len(files), face_index), end="")
        
    print()

def find_from_vedio():

    frame_skip = 10
    input_path = "/home/repositories/test-data/faces/video/2-5m.mp4"
    result_path = os.path.join("/home/repositories/test-data/", "faces/face_result_v/file_2")

    # Open video file
    vcap = cv2.VideoCapture(input_path)
    frame_index = 0
    face_index = 0

    # get base info from input video
    while vcap.isOpened():
        w  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fps = vcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        break
        
    print('width, height:', w, h)
    print('fps:', fps)  # float
    print('frames count:', frame_count)  # float

    #frame_in_face = 0
    next_frame = frame_skip
    while vcap.isOpened():

        # Grab a single frame of video
        ret, frame = vcap.read()
        # Bail out when the video file ends
        if not ret:
            break

        frame_index += 1

        # skip frame to next frame
        if divmod(frame_index, next_frame)[1] != 0:
            continue

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # 缩小图片来寻找人脸，更大的数值寻找更小的人脸。
        scale = 200 / w
        sw, sh = int(scale * w), int(scale * h)
        small_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_CUBIC)
        #small_img = org_img.resize((sw, sh), Image.ANTIALIAS).convert('RGB')

        face_locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample=1)
        #face_encodings = face_recognition.face_encodings(frame, face_locations)
        if len(face_locations) == 0:
            next_frame = frame_skip
        else:
            # do not get face immediately. skip 10 frames.
            next_frame = 10

        # Loop through each face found in the unknown image
        for top, right, bottom, left in face_locations:

            #frame_in_face += 1
            #if divmod(frame_in_face, 10)[1] != 0:
            #    continue

            # do not get face if size < 200 * scale pixel
            if right - left < int(200 * scale):
                #frame_in_face = 0 # init face frame to continue find next face
                continue

            # 截取的面部范围扩大10%
            inc_pix = (right - left) * 0.1

            # increate located face size 
            top = int((top - inc_pix) / scale)
            if top < 0: top = 0
            left = int((left - inc_pix) / scale)
            if left < 0: left = 0

            bottom = int((bottom + inc_pix) / scale)
            if bottom > h: bottom = h
            right = int((right + inc_pix) / scale)
            if right > w: right = w

            #frame_in_face = 0 # init face frame to continue find next face
            #print("    get face in frame #%s" % (frame_index), end='')

            # You can access the actual face itself like this:
            face_image = frame[top:bottom, left:right]
            pil_face_image = Image.fromarray(face_image)

            # You can also save a copy of the new image to disk if you want by uncommenting this line
            save_path = os.path.join(result_path, ("%s.jpg" % (frame_index)))
            pil_face_image.save(save_path)

            face_index += 1
        
        print("\r %s/%s frames completed. %d faces saved." % (frame_index, frame_count, face_index), end="")

    print()   
    #print("Parse Completed.Found %s faces" % (face_index))


find_from_vedio()
#find_from_image()
