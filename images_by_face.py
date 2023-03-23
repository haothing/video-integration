'''
从图片或者视频中截取人物。
可以根据输入的脸部信息截取，或者截取全部的人物。
'''
import argparse
import os

import cv2
# import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import ToTensor

# python get_faces.py --input_image_path E:\datasets\manga\mp4_screen\1 --save_path E:\datasets\manga\mp4_face\finded
# python get_faces.py --input_video_path E:\datasets\faces\video\2-5m.mp4 --target_face data/target_face
parser = argparse.ArgumentParser()
parser.add_argument("--input_image_path", type=str, default='', help="Image file or directory, where to find faces")
parser.add_argument("--input_video_path", type=str, default='', help="Video file or directory, where to find faces")
parser.add_argument("--save_path", type=str, default="data/result", help="Path to save result faces file")
parser.add_argument("--target_face_folder", type=str, default='data/target_face/isihara',
    help="The target face to be obtained, when specifying a folder, all files in the folder are the target, all faces get when has't this parameter.")
parser.add_argument("--min_face_size", type=int, default=200, help="The min face size in image, both width or height.")
parser.add_argument("--face_threshold", type=float, default=0.72, help="The face confidence threshold.Min value to find similar face.")
# parser.add_argument("--target_gender", type=str, default='', help="The face gender(woman or man) to find, all faces get when blank.")
# parser.add_argument("--target_emotion", type=str, default='', help="The face emotion to find, all faces get when blank. String split by ',' can be 'angry,disgust,fear,happy,sad,surprise,neutral'")
parser.add_argument("--frame_interval", type=int, default=10, help="The interval frame number, skip the frame number to get the image from the video.")
opt = parser.parse_args()

YOLO_PERSON_CLASS_CODE = 0
YOLO_PERSON_CONFIDENCE_THRESHOLD = 0.8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

modelYolo = YOLO('yolov8n.pt')
mtcnn = MTCNN(
    image_size=200, margin=0,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device, keep_all=False, select_largest=True
)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

def get_target_emb(target_face_folder):
    '''
    从要识别的目标人脸中取得映射空间，根据此数据同输入人脸进行比较。
    '''
    images = []
    # Loop through all the files in the folder
    for filename in os.listdir(target_face_folder):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Open the image using PIL and add it to the list
            image_path = os.path.join(target_face_folder, filename)
            image = Image.open(image_path)
            images.append(image)

    assert images is not None and len(images) > 0, "Can't find image in the target folder."
    # Get aliogned face in images by mtcnn.
    aligned = mtcnn(images)
    assert aligned[0] is not None, "Can't find face in the target folder."
    print(f"Target face name {os.path.dirname(target_face_folder)}, face count: {len(aligned)}.")

    # Get embeddings from aligned face by resnet(InceptionResnetV1).
    aligned = torch.stack(aligned).to(device)
    return resnet(aligned).detach().cpu()

def compare_emb(target_embs, input_emb, face_threshold):
    '''
    target_embs为对象人脸映射值数组，对象人脸和input_emb进行匹配，
    相似度超过阈值时返回True，否则返回False。
    '''
    compare_point = 0
    for t_emb in target_embs:
        compare_point += (t_emb - input_emb).norm().item()
    return compare_point / len(target_embs) <= face_threshold

def find_from_vedio(input_path, save_path, target_face_folder, face_threshold=0.8, min_face_size=200):
    '''
    从视频中截取人物。
    可以根据输入的脸部信息截取，或者截取全部的人物。只支持单一人脸。
    '''
    target_emb = get_target_emb(target_face_folder)
    face_name = os.path.basename(target_face_folder)
    # 预先建立输出文件夹
    save_path = os.path.join(save_path, face_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 不是目录时，只处理单一文件
    files = []
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
        files.sort()
        # change files to full path
        files = [os.path.join(input_path, f) for f in files]
        # sort files by name in int
    else:
        assert os.path.isfile(input_path), "Can't find input file."
        files.append(input_path)

    # 遍历所有文件
    for file in files:
        # skip sub dirs
        if os.path.isdir(file):
            continue
        # Open video file
        vcap = cv2.VideoCapture(file)
        file_name_v, frame_index, face_index = os.path.splitext(os.path.basename(file))[0], 0, 0
        while vcap.isOpened():
            frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_wh = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "/" + int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            break

        while vcap.isOpened():
            # Grab a single frame of video
            ret, frame = vcap.read()
            # Bail out when the video file ends
            if not ret:
                break
            frame_index += 1
            # skip frame to next frame
            if divmod(frame_index, opt.frame_interval)[1] != 0:
                continue
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            # find person by yolo
            results = modelYolo.predict(source=img, verbose=False)  # predict on an image
            # only crop person box in frame
            images, result = [], results[0]
            for idx, cls in enumerate(result.boxes.cls):
                if cls.int().item() == YOLO_PERSON_CLASS_CODE and result.boxes.conf[idx] > YOLO_PERSON_CONFIDENCE_THRESHOLD:
                    result.boxes.xyxy[idx].tolist()
                    cropped_img  = img.crop(result.boxes.xyxy[idx].tolist())
                    images.append(cropped_img)

            # Get face from cropped images by mtcnn.
            for img in images:
                boxes, probs, _ = mtcnn.detect(img, landmarks=True)
                # when face found in image and face size is larger than threshold, crop face and save it.
                if boxes is not None and\
                    (boxes[0][2] - boxes[0][0] > min_face_size or boxes[0][3] - boxes[0][1] > min_face_size) and\
                    probs[0] > 0.999:
                    # Crop face and resize it to 200x200 and convert to tensor.
                    aligned = img.crop(boxes[0].tolist()).resize((200, 200), Image.BILINEAR)
                    aligned = ToTensor()(aligned)

                    # Get embeddings from aligned face by resnet(InceptionResnetV1).
                    input_emb = resnet(torch.unsqueeze(aligned, 0).to(device)).detach().cpu()
                    # Compare target face and input face, output cropped images when threshold reached.
                    if compare_emb(target_emb, input_emb, face_threshold):
                        file_name = f"{file_name_v}_{face_index}_f({frame_index}).jpg"
                        img.save(os.path.join(save_path, file_name))
                        face_index += 1
            print(f"\r  Input file: {file}, width/height: {frame_wh}, {frame_index}/{frame_count} frames completed. {face_index} images saved.", end="")
        print()

if __name__ == '__main__':

    # if opt.input_image_path != '':
    #     find_from_image(opt.input_image_path)

    if opt.input_video_path != '':
        find_from_vedio(opt.input_video_path,
                        save_path = opt.save_path,
                        target_face_folder = opt.target_face_folder,
                        face_threshold = opt.face_threshold,
                        min_face_size = opt.min_face_size)
