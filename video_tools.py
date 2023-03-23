'''
各种视频处理工具
'''
import argparse
import os

import moviepy.editor as movpy

parser = argparse.ArgumentParser()
parser.add_argument("function_name", type=str, default='mkv2mp4', help="Which functin to be execute.")
parser.add_argument("-i", "--input_video_path", type=str, default='', help="Input video file.")
opt = parser.parse_args()

print(opt)
# get image from video
# offset_t: skip offset_t time to start get images
# interval = get image every time in seconds
# def get_image_from_video(video, offset_t = 0, interval = 300):

#     # get the end time
#     for time_index in range(int(video.duration - offset_t), int(video.duration), interval):

#         #plt.subplot(10, 1, imageIndex)
#         #print(frameIndex)
#         f = video.get_frame(time_index)

#         img = Image.fromarray(f, 'RGB')
#         img.save('../data/%s_%d.jpg' % (vedio.file_name, time_index))
#         print('/%s_%d.jpg.jpg saved' % (vedio.file_name, time_index))

#     return end_time

def mkv2mp4(input_mkv):
    '''
    转换mkv格式视频为mp4格式。
    Args:
        input(file): mkv格式输入文件
        
    Returns:
        output(file): mp4格式输出文件，和输入文件同一文件夹。
    '''
    if os.path.isdir(input_mkv):
        mkv2mp4_folder(input_mkv)
        return

    folder = os.path.dirname(input_mkv)
    file_name = os.path.basename(input_mkv)
    pre, _ = os.path.splitext(file_name)

    clip = movpy.VideoFileClip(input_mkv)
    clip.write_videofile(os.path.join(folder, pre + '.mp4'), codec="libx264",audio_codec="aac")

def mkv2mp4_folder(input_folder):
    '''
    转换mkv格式视频为mp4格式。
    Args:
        input_folder(folder): mkv格式输入文件夹
        
    Returns:
        output_folder(folder): mp4格式输出文件夹，和输入文件夹同一文件夹。
    '''
    for file_name in os.listdir(input_folder):
        _, ext = os.path.splitext(file_name)
        if ext == '.mkv':
            mkv2mp4(os.path.join(input_folder, file_name))
        
if __name__ == '__main__':

    # python video_tools.py mkv2mp4 -i "E:/datasets/faces/video/2/Jimi ni Sugoi! EP01 720p HDTV x264 AAC-DoA.mkv"
    if opt.function_name == 'mkv2mp4':
        mkv2mp4(opt.input_video_path)
