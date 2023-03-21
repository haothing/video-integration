import os
import argparse 
import time

import numpy as np
from PIL import Image
 
import cv2
import torch
from torchvision.transforms import ToPILImage
 
# import face_recognition
# from face_information import FaceInformation
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
 
# python get_faces.py --input_image_path E:\datasets\manga\mp4_screen\1 --save_path E:\datasets\manga\mp4_face\finded
# python get_faces.py --input_video_path E:\datasets\faces\video\2-5m.mp4 --target_face data/target_face
parser = argparse.ArgumentParser()
parser.add_argument("--input_image_path", type=str, default='', help="Image file or directory, where to find faces")
parser.add_argument("--input_video_path", type=str, default='', help="Video file or directory, where to find faces")
parser.add_argument("--save_path", type=str, default="data/result", help="Path to save result faces file")
parser.add_argument("--target_face", type=str, default='data/target_face',
    help="The target face to be obtained, when specifying a folder, all files in the folder are the target, all faces get when has't this parameter.")
parser.add_argument("--face_threshold", type=float, default=0.32, help="Face threshold to find known faces, small value to get less faces.")
# parser.add_argument("--target_gender", type=str, default='', help="The face gender(woman or man) to find, all faces get when blank.")
# parser.add_argument("--target_emotion", type=str, default='', help="The face emotion to find, all faces get when blank. String split by ',' can be 'angry,disgust,fear,happy,sad,surprise,neutral'")
parser.add_argument("--frame_interval", type=int, default=5, help="The interval frame number, skip the frame number to get the image from the video.")
opt = parser.parse_args()

# if opt.target_emotion != '': 
#     opt.target_emotion = opt.target_emotion.split(',')

print(opt)

def find_from_image(input_path):

    result_path = opt.save_path
    
    files = []
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
    else:
        files.append(input_path)

    # 按文件名排序
    files.sort()
    
    # 遍历所有文件
    file_index = 0
    face_index = 0
    for file in files:

        # skip sub dirs
        if os.path.isdir(os.path.join(input_path, file)):
            continue

        # 如果后缀名为.jpg或.png
        ext = os.path.splitext(file)[1].lower()
        if ext in [".jpg", ".png"]:
            # 拼接成完整路径
            filePath = os.path.join(input_path, file)

        # load image file
        org_img = Image.open(filePath)
        w, h = org_img.size

        # 缩小图片来寻找人脸，更大的数值寻找更小的人脸。
        scale = 800 / w
        sw, sh = int(scale * w), int(scale * h)
        small_img = org_img.resize((sw, sh), Image.ANTIALIAS).convert('RGB')

        face_locations = face_recognition.face_locations(np.array(small_img), number_of_times_to_upsample=1, model='cnn')
        # Loop through each face found in the unknown image
        for top, right, bottom, left in face_locations:
            
            # get expand pix 
            x_offset = 0
            # y_offset = (right - left) * 0.1
            y_offset = (right - left) * 0.1

            expand = (right - left) * 0.05
            # increate located face size 
            top = int((top - y_offset - expand) / scale)
            if top < 0: top = 0
            left = int((left - x_offset - expand) / scale)
            if left < 0: left = 0

            bottom = int((bottom - y_offset + expand) / scale)
            if bottom > h: bottom = h
            right = int((right - x_offset + expand) / scale)
            if right > w: right = w

            # access the actual face(RGB image)
            face_image = org_img.crop((left, top, right, bottom))

            # You can also save a copy of the new image to disk if you want by uncommenting this line
            save_path = os.path.join(result_path, ("%s_%s.jpg" % (os.path.splitext(file)[0], face_index)))
            face_image.save(save_path)
            face_index += 1

        file_index += 1
        print("\r %d/%d files completed. %d faces saved." % (file_index, len(files), face_index), end="")
        
    print()

def find_from_vedio_face_recognition(input_path):

    frame_skip = 5
    face_name = 'all'
    result_path = opt.save_path

    if opt.target_face != '':
        face_info = FaceInformation()
        face_info.load_known_faces(opt.target_face)
        print(face_info.known_face_names)
    
    # 不是目录时，只处理单一文件
    files = []
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
    else:
        files.append(input_path)

    # 按文件名排序
    files.sort()

    # 遍历所有文件
    for file in files:

        full_path = os.path.join(input_path, file)

        # skip sub dirs
        if os.path.isdir(full_path):
            continue

        # Open video file
        vcap = cv2.VideoCapture(full_path)
        frame_index = 0
        face_index = 0

        # get base info from input video
        while vcap.isOpened():
            w  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
            fps = vcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            break

        print("Start with file: %s" % (file))
        print('--width, height:', w, h)
        print('--fps:', fps)  # float
        print('--frames count:', frame_count)  # float

        #frame_in_face = 0
        next_frame = opt.frame_interval
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
            small_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
            #small_img = org_img.resize((sw, sh), Image.ANTIALIAS).convert('RGB')

            face_locations = face_recognition.face_locations(small_frame, model='cnn')

            #face_encodings = face_recognition.face_encodings(frame, face_locations)
            if len(face_locations) == 0:
                next_frame = opt.frame_interval
            else:
                # 跳过一定帧数取得下一张脸部图像，避免取得相似的图像
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

            # get expand pix 
                x_offset = 0
                # y_offset = (right - left) * 0.1
                y_offset = (right - left) * 0.1

                expand = (right - left) * 0.05
                # increate located face size 
                top = int((top - y_offset - expand) / scale)
                if top < 0: top = 0
                left = int((left - x_offset - expand) / scale)
                if left < 0: left = 0

                bottom = int((bottom - y_offset + expand) / scale)
                if bottom > h: bottom = h
                right = int((right - x_offset + expand) / scale)
                if right > w: right = w

                #frame_in_face = 0 # init face frame to continue find next face
                #print("    get face in frame #%s" % (frame_index), end='')

                # You can access the actual face itself like this: (RGB image)
                face_image = frame[top:bottom, left:right]
                #gender, emotion = 1, 1

                if opt.target_face != '':
                    face_name, face_distance = face_info.get_face_name(face_image)
                    
                    print(face_name, face_distance)
                    if face_name not in face_info.known_face_names:
                        continue
                    if face_distance > opt.face_threshold: 
                        continue
                
                # if opt.target_gender != '' or opt.target_emotion != '':
                #     gender, emotion = face_info.get_base_info(face_image)
                #     if gender != opt.target_gender:
                #         continue
                #     if emotion not in opt.target_emotion:
                #         continue

                pil_face_image = Image.fromarray(face_image)
                pil_face_image = pil_face_image.resize((256, 256),Image.ANTIALIAS)
                # You can also save a copy of the new image to disk if you want by uncommenting this line
                # save_path = os.path.join(result_path, ("%s_%s_%s_%s.jpg" % (face_name, gender, emotion, frame_index)))
                save_path = os.path.join(result_path, face_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                file_name = ("%s.jpg" % (frame_index))
                pil_face_image.save(os.path.join(save_path, file_name))

                face_index += 1
            
            print("\r    -- %s/%s frames completed. %d faces saved." % (frame_index, frame_count, face_index), end="")

        print()   
        #print("Parse Completed.Found %s faces" % (face_index))

def find_from_vedio(input_path):

    frame_skip = 5
    face_name = 'all'
    result_path = opt.save_path

    # if opt.target_face != '':
    #     face_info = FaceInformation()
    #     face_info.load_known_faces(opt.target_face)
    #     print(face_info.known_face_names)
    
    # 不是目录时，只处理单一文件
    files = []
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
    else:
        files.append(input_path)
    # 按文件名排序
    files.sort()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=False
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 遍历所有文件
    for file in files:
        full_path = os.path.join(input_path, file)
        # skip sub dirs
        if os.path.isdir(full_path):
            continue

        # Open video file
        vcap = cv2.VideoCapture(full_path)
        frame_index = 0
        face_index = 0
        # get base info from input video
        while vcap.isOpened():
            w  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
            fps = vcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            break

        print("Start with file: %s" % (file))
        print('--width, height:', w, h)
        print('--fps:', fps)  # float
        print('--frames count:', frame_count)  # float

        #frame_in_face = 0
        next_frame = opt.frame_interval
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # # 缩小图片来寻找人脸，更大的数值寻找更小的人脸。
            # scale = 200 / w
            # sw, sh = int(scale * w), int(scale * h)
            # small_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
            # #small_img = org_img.resize((sw, sh), Image.ANTIALIAS).convert('RGB')

            # face_locations = face_recognition.face_locations(small_frame, model='cnn')
            img = Image.fromarray(frame)

            # find person by yolo
            model = YOLO('yolov8n.pt')
            results = model.predict(source=img, verbose=False)  # predict on an image
            # TODO only get person in result and handel muitle persons
            
            # im1 = Image.open(target_file)
            # results = model.predict(source=im1, save=False)  # save plotted images

            # get plot in ndarray, WHC formate
            res_plotted = results[0].plot()
            # cut person from detected results
            x, y, x1, y1 = results[0].boxes.xyxy[0].tolist()
            cropped_img  = img.crop((x, y, x1, y1))
            # find facial in person cuted.            
            aligned, prob = mtcnn(cropped_img, return_prob=True)
            if aligned is None:
                continue
            print(prob)
            to_pil = ToPILImage()
            aligned = to_pil((aligned + 1) / 2)
            aligned.show()
            
            # res_plotted = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(np.asarray(res_plotted))
            return
            
            #face_encodings = face_recognition.face_encodings(frame, face_locations)
            if len(face_locations) == 0:
                next_frame = opt.frame_interval
            else:
                # 跳过一定帧数取得下一张脸部图像，避免取得相似的图像
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

            # get expand pix 
                x_offset = 0
                # y_offset = (right - left) * 0.1
                y_offset = (right - left) * 0.1

                expand = (right - left) * 0.05
                # increate located face size 
                top = int((top - y_offset - expand) / scale)
                if top < 0: top = 0
                left = int((left - x_offset - expand) / scale)
                if left < 0: left = 0

                bottom = int((bottom - y_offset + expand) / scale)
                if bottom > h: bottom = h
                right = int((right - x_offset + expand) / scale)
                if right > w: right = w

                #frame_in_face = 0 # init face frame to continue find next face
                #print("    get face in frame #%s" % (frame_index), end='')

                # You can access the actual face itself like this: (RGB image)
                face_image = frame[top:bottom, left:right]
                #gender, emotion = 1, 1

                if opt.target_face != '':
                    face_name, face_distance = face_info.get_face_name(face_image)
                    
                    print(face_name, face_distance)
                    if face_name not in face_info.known_face_names:
                        continue
                    if face_distance > opt.face_threshold: 
                        continue
                
                # if opt.target_gender != '' or opt.target_emotion != '':
                #     gender, emotion = face_info.get_base_info(face_image)
                #     if gender != opt.target_gender:
                #         continue
                #     if emotion not in opt.target_emotion:
                #         continue

                pil_face_image = Image.fromarray(face_image)
                pil_face_image = pil_face_image.resize((256, 256),Image.ANTIALIAS)
                # You can also save a copy of the new image to disk if you want by uncommenting this line
                # save_path = os.path.join(result_path, ("%s_%s_%s_%s.jpg" % (face_name, gender, emotion, frame_index)))
                save_path = os.path.join(result_path, face_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                file_name = ("%s.jpg" % (frame_index))
                pil_face_image.save(os.path.join(save_path, file_name))

                face_index += 1
            
            print("\r    -- %s/%s frames completed. %d faces saved." % (frame_index, frame_count, face_index), end="")

        print()   
        #print("Parse Completed.Found %s faces" % (face_index))
        
if __name__ == '__main__':

    if opt.input_image_path != '':
        find_from_image(opt.input_image_path)
    
    if opt.input_video_path != '':
        find_from_vedio(opt.input_video_path)
