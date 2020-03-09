import os
import numpy as np

from PIL import Image, ImageDraw
import face_recognition



'''
picture_of_me = face_recognition.load_image_file("/mnt/hgfs/ubuntu_ml_shared/face-recognition/known.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("/mnt/hgfs/ubuntu_ml_shared/face-recognition/1.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

# results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
# results = face_recognition.face_distance([my_face_encoding], unknown_face_encoding)

print(results)
'''

file_index = 0
path = "/mnt/hgfs/ubuntu_ml_shared/face-recognition/"
for root, dirs, files in os.walk(path):
  
  # skip sub dirs
  if (path != root):
    continue
  
  # 按文件名排序
  files.sort()

  # 遍历所有文件
  for file in files:

    # 如果后缀名为.jpg
    if os.path.splitext(file)[1] == '.jpg':
      # 拼接成完整路径
      filePath = os.path.join(root, file)

      # load file 
      print("file %s is processing." % (filePath))
      image = face_recognition.load_image_file(filePath)
      face_locations = face_recognition.face_locations(image)

      # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
      pil_image = Image.fromarray(image)
      # Create a Pillow ImageDraw Draw instance to draw with
      draw = ImageDraw.Draw(pil_image)

      for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location

        top = top - 100
        if top < 0: top = 0
        left = left - 20
        if left < 0: left = 0

        bottom = bottom + 20
        if bottom > pil_image.height: bottom = pil_image.height
        right = right + 20
        if right > pil_image.width: right = pil_image.width

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=4)

        # You can also save a copy of the new image to disk if you want by uncommenting this line
        pil_image.save("/mnt/hgfs/ubuntu_ml_shared/face-recognition/result/%s" % file)
            
        file_index += 1

# Remove the drawing library from memory as per the Pillow docs
del draw

print('%d files processed.' % file_index)




