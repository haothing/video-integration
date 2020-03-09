#coding=utf-8
import os
import numpy as np

from PIL import Image, ImageDraw
import face_recognition

#known_path = os.path.join("/home/appuser/repositories/video-integration/", "data/known/")
#unknown_path = os.path.join("/home/appuser/repositories/video-integration/", "data/1/")

known_path = os.path.join("/home/repositories/test-data/", "faces/known/")
unknown_path = os.path.join("/home/repositories/test-data/", "faces/face_image/")
result_path = os.path.join("/home/repositories/test-data/", "faces/face_result/")

#face_recognition.load_image_file("../data/test/1.jpg")
#print("dir:" + os.path.dirname(__file__) )
#print("dir:" + os.path.dirname(os.path.dirname(__file__)))

# Load a sample picture and learn how to recognize it.
k1_image = face_recognition.load_image_file(os.path.join(known_path, "known_1.jpg"))
k1_face_encoding = face_recognition.face_encodings(k1_image)[0]

# Load a second sample picture and learn how to recognize it.
k2_image = face_recognition.load_image_file(os.path.join(known_path, "known_2.jpg"))
k2_face_encoding = face_recognition.face_encodings(k2_image)[0]

# Load a second sample picture and learn how to recognize it.
k3_image = face_recognition.load_image_file(os.path.join(known_path, "known_3.jpg"))
k3_face_encoding = face_recognition.face_encodings(k3_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    k1_face_encoding,
    k2_face_encoding,
    k3_face_encoding
]
known_face_names = [
    "SUN LILI",
    "SUN JIANJUN",
    "SUN QIANYI"
]

file_index = 0
unknown_images = []

for root, dirs, files in os.walk(unknown_path):
  

  # skip sub dirs
  if (unknown_path != root):
    continue
  
  # 按文件名排序
  files.sort()

  # 遍历所有文件
  for file in files:

    # 如果后缀名为.jpg
    if os.path.splitext(file)[1].lower() == '.jpg':
      # 拼接成完整路径
      filePath = os.path.join(root, file)

      # load file 
      print("file %s is processing." % (filePath))
      unknown_images.append(face_recognition.load_image_file(filePath))
    
  face_locations = face_recognition.batch_face_locations(unknown_images, number_of_times_to_upsample=1, batch_size=2)

  print(face_locations)

'''
    face_encodings = face_recognition.face_encodings(unknown_images, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    # draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

      # See if the face is a match for the known face(s)
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
      
      name = "Unknown"

      # If a match was found in known_face_encodings, just use the first one.
      # if True in matches:
      #   first_match_index = matches.index(True)
      #   name = known_face_names[first_match_index]

      # Or instead, use the known face with the smallest distance to the new face
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]

      print("    face_distances: %s. name: %s." % (np.fromstring(face_distances.tostring(), dtype=int), name))

      top = top - 100
      if top < 0: top = 0
      left = left - 20
      if left < 0: left = 0

      bottom = bottom + 20
      if bottom > pil_image.height: bottom = pil_image.height
      right = right + 20
      if right > pil_image.width: right = pil_image.width

      # You can access the actual face itself like this:
      face_image = unknown_image[top:bottom, left:right]
      pil_face_image = Image.fromarray(face_image)

      # You can also save a copy of the new image to disk if you want by uncommenting this line
      save_path = os.path.join(result_path, ("%s_%s.jpg" % (os.path.splitext(file)[0], name)))
      pil_face_image.save(save_path)

      file_index += 1
'''
# Remove the drawing library from memory as per the Pillow docs
# del draw

print('%d faces processed.' % file_index)






