'''
FaceInformation Class,整合面部信息，包括表情，性别，人物。
'''
import os
import cv2
# from keras.models import load_model
import numpy as np

import face_recognition
from face_classification.src.utils.datasets import get_labels
from face_classification.src.utils.preprocessor import preprocess_input

class FaceInformation:
    '''
    FaceInformation Class，整合面部信息，包括表情，性别，人物。
    '''
    def __init__(self):

        # 定义已知的人脸信息
        self.known_face_names = []
        self.known_face_encodings = []


    def load_known_faces(self, input_path="/"):
        '''
        装载input_path目录中所有的已知人脸，如果input_path是文件则只装载该文件
        文件名为默认的人物姓名
        '''
        if not os.path.isdir(input_path):
            files= input_path
        else:
            files= os.listdir(input_path)
        
        for file in files:

            # 拼接成完整路径
            file_path = os.path.join(input_path, file)

            # skip sub dirs
            if os.path.isdir(file_path):
                continue

            # 如果后缀名为.jpg或.png
            face_name = os.path.splitext(file)[0].lower()
            ext = os.path.splitext(file)[1].lower()
            if ext not in [".jpg", ".jpeg"," .png"]:
                continue

            # 读取人脸图像
            face_image = face_recognition.load_image_file(file_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_names.append(face_name)
            self.known_face_encodings.append(face_encoding)


    def get_face_name(self, rgb_face_array):
        '''
        识别人物姓名，需要预先调用load_known_faces装载已知的人物面部图像
        返回人物姓名
        '''
        assert len(self.known_face_names) > 0, "You must excute load_known_faces method to load known faces first."
        # use the known face with the smallest distance to the new face
        #face_distances = face_recognition.face_distance(self.known_face_encodings, rgb_face_array)
        #best_match_index = np.argmin(face_distances)
        #if matches[best_match_index]:
        #   name = known_face_names[best_match_index]

        name = "x"
        codes = face_recognition.face_encodings(rgb_face_array)
        if not len(codes) > 0:
            return name, 1

        face_encoding = codes[0]
        # tolerance – How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        #if True in matches:
        #   first_match_index = matches.index(True)
        #   name = self.known_face_names[first_match_index]

        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
        #else:
        #   return name, 1

        #print(face_distances)
        return name, face_distances[best_match_index]
