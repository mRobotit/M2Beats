import os

from moviepy.editor import *
from tqdm import tqdm

asset_path = r"./MOVIE_ASSET/movie"

def change_speed(video, acc_list,acc_rate=3):
    """
    改变视频的速度，每个分块的前1/4段加速，中间减速，后1/4段加速

    [MoviePy clip相关的重要api](https://juejin.im/post/5d1c4318f265da1ba9159912)
    :param video_path:视频路径
    :param speed:速度
    :param start:开始时间
    :param end:结束时间
    :return:
    """
    os.makedirs(asset_path, exist_ok=True)

    
    video_list = []
    i=0
    for acc in acc_list:
        start = acc[0]
        end = acc[1]
        target_time = acc[2]
        speed = (end - start) / target_time

        clip1_start = start
        clip1_end = start + (end-start)*(6/8)
        clip1_speed = (clip1_end-clip1_start)/(target_time*(6/8)) * acc_rate
        clip1 = video.subclip(clip1_start, clip1_end)
        clip1 = clip1.fl_time(lambda t: clip1_speed * t, apply_to=['mask', 'video', 'audio']).set_end((clip1_end-clip1_start) / clip1_speed)
        # clip1_path = os.path.join(asset_path,"{}.mp4".format(i))
        # i+=1
        # clip1.write_videofile(clip1_path)

        clip2_start = clip1_end
        clip2_end = end
        clip2_speed = (clip2_end - clip2_start) / (target_time - (target_time*(6/8) / acc_rate))
        clip2 = video.subclip(clip2_start, clip2_end)
        clip2 = clip2.fl_time(lambda t: clip2_speed * t, apply_to=['mask', 'video', 'audio']).set_end((clip2_end - clip2_start) / clip2_speed)
        # clip2_path = os.path.join(asset_path,"{}.mp4".format(i))
        # i+=1
        # clip2.write_videofile(clip2_path)

        # clip3_start = clip2_end
        # clip3_end = end
        # clip3_speed = (clip3_end-clip3_start)/(target_time*(1/8)) * acc_rate
        # clip3 = video.subclip(clip3_start, clip3_end)
        # clip3 = clip3.fl_time(lambda t: clip3_speed * t, apply_to=['mask', 'video', 'audio']).set_end((clip3_end - clip3_start) / clip3_speed)
        

        change_part = concatenate_videoclips([clip1,clip2], method='chain')
        path = os.path.join(asset_path,"{}.mp4".format(i))
        i+=1
        change_part.write_videofile(path)

        
    if end < video.end - 0.1:
        end_part = video.subclip(end)
        path = os.path.join(asset_path,"{}.mp4".format(i))
        i+=1
        end_part.write_videofile(path)
    # video_list.append(end_part)

    # 直接用这个会出问题
    # result_video = concatenate_videoclips(video_list, method='chain')

    n = i
    i=0
    while i<n:
        path = os.path.join(asset_path, "{}.mp4".format(i))
        video_part = VideoFileClip(path)
        video_list.append(video_part)
        i+=1
    result_video = concatenate_videoclips(video_list, method='chain')

    return result_video

# def change_speed(video, acc_list):
#     os.makedirs(asset_path, exist_ok=True)
#     video_list = []
#     i=0
#     for acc in acc_list:
#         start = acc[0]
#         end = acc[1]
#         target_time = acc[2]
#         speed = (end - start) / target_time
#         change_part = video.subclip(start, end)
#         change_part = change_part.fl_time(lambda t: speed * t, apply_to=['mask', 'video', 'audio']).set_end((end - start) / speed)
#         # video_list.append(change_part)

#         path = os.path.join(asset_path,"{}.mp4".format(i))
#         i+=1
#         change_part.write_videofile(path)
        
#     if end < video.end - 0.1:
#         end_part = video.subclip(end)
#         i+=1
#         path = os.path.join(asset_path,"{}.mp4".format(i))
#         end_part.write_videofile(path)
#     # video_list.append(end_part)

#     # 直接用这个会出问题
#     # result_video = concatenate_videoclips(video_list, method='chain')

#     n = i+1
#     i=0
#     while i<n:
#         path = os.path.join(asset_path, "{}.mp4".format(i))
#         video_part = VideoFileClip(path)
#         video_list.append(video_part)
#         i+=1
#     result_video = concatenate_videoclips(video_list, method='chain')

#     return result_video


if __name__ == '__main__':
    movie_path = r"/data/jdx/code/mycode/movie/xinxiaomeng_small.mp4"

    video = VideoFileClip(movie_path)
    acc_list = [
        [0, 2, 4],
        [2, 8, 3]
    ]

    result_video = change_speed(video, acc_list)

    result_path = './result.mp4'
    result_video.write_videofile(result_path)
