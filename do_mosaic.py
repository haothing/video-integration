import cv2
##马赛克
def do_mosaic(frame, x, y, w, h, neighbor=9):
  """
  马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
  :param frame: opencv frame
  :param int x : 马赛克左顶点
  :param int y: 马赛克右顶点
  :param int w: 马赛克宽
  :param int h: 马赛克高
  :param int neighbor: 马赛克每一块的宽
  """
  fh, fw = frame.shape[0], frame.shape[1]
  if (y + h > fh) or (x + w > fw):
    return
  for i in range(0, h - neighbor, neighbor): # 关键点0 减去neightbour 防止溢出
    for j in range(0, w - neighbor, neighbor):
      rect = [j + x, i + y, neighbor, neighbor]
      color = frame[i + y][j + x].tolist() # 关键点1 tolist
      left_up = (rect[0], rect[1])
      right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1) # 关键点2 减去一个像素
      cv2.rectangle(frame, left_up, right_down, color, -1)
im = cv2.imread('test.jpg', 1)
do_mosaic(im, 219, 61, 460 - 219, 412 - 61)
 
while 1:
  k = cv2.waitKey(10)
  if k == 27:
    break
  cv2.imshow('mosaic', im)
