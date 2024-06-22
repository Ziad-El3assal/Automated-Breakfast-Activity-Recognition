import av
import PIL
import skimage.io
from skimage.transform import resize, pyramid_reduce

import numpy as np
import os
import argparse

"""
NOTES:
"PyAV is a Pythonic binding for FFmpeg" (https://mikeboers.github.io/PyAV/).
This script was tested with a 1920x1080 video. Crop and resize sizes should be changed in the method below as per 
your requirements.
"""


def video_frames_writer(video_dir, video_name, writer_type, crop_and_resize_image,sv_dir=None):
    """
    Writes a (height, width, 3) shaped image in a subdirectory named after the video's name in the video_dir.
    :param video_dir: directory where the video is located
    :param video_name: .avi video file name
    :param writer_type: package used for writing the frames onto disk; 0: PIL; 1: skimage
    :param crop_and_resize_image: boolean flag
    :return: None
    """
    #Breakfast Action Recognition\train\P03\P03_cereals.avi
    save_dir = video_dir + video_name.split('.')[0]
    sv_dir=os.path.join(sv_dir,video_name.split('.')[0])
    os.makedirs(os.path.join('.', sv_dir), exist_ok=True)
    if not os.path.exists(sv_dir) or len(os.listdir(sv_dir))==0:
        os.makedirs(sv_dir,exist_ok=True)
    else:
        print(f"Folder {sv_dir} already exists")
        return
    container = av.open(os.path.join(video_dir , video_name), metadata_errors='ignore')
    for frame in container.decode(video=0):
        if frame.index % 100 == 0:
            
            print("processed frame index {}".format(frame.index))

        img_pil = frame.to_image()
        width, height = img_pil.size  # width, height for this read PIL image

        if writer_type == 0:
            if crop_and_resize_image:
                # 0 crop and resize using PIL
                img_pil_cropped = img_pil.crop((240, 0, width - 240, height))  # (x0, y0, x1, y1)
                resize_width = 320
                ratio = resize_width / float(img_pil_cropped.size[0])
                resize_height = int(float(img_pil_cropped.size[1]) * float(ratio))
                img_pil_down = img_pil_cropped.resize((resize_width, resize_height), PIL.Image.ANTIALIAS)
                img_pil_down.save(os.path.join(sv_dir, video_name.split('.')[0]) + '_frame_%05d.jpg' % frame.index)
            else:
                img_pil.save(os.path.join(sv_dir, video_name.split('.')[0]) + '_frame_%05d.jpg' % frame.index)

        elif writer_type == 1:
            img_arr = np.asarray(img_pil)  # converting PIL (<class 'PIL.Image.Image'>) to (<class 'numpy.ndarray'>)
            h, w, c = img_arr.shape  # h, w, c for skimage
            # this is exact same as when you read a .jpg image using skimage.io.imread (h, w, c)

            if crop_and_resize_image:
                # 1 crop and resize using skimage.io
                print(h,w)
                img_sk_cropped = img_arr[0:h, 240:w-240]  # x0:x1,y0:y1
                img_sk_down = pyramid_reduce(img_sk_cropped, downscale=4.5)
                skimage.io.imsave(os.path.join(sv_dir, video_name.split('.')[0]) + '_frame_%05d.jpg' % frame.index,
                                  img_sk_down)
            else:
                skimage.io.imsave(os.path.join(sv_dir, video_name.split('.')[0]) + '_frame_%05d.jpg' % frame.index,
                                  img_arr)

    # container.close()
    print("Processed the video {} and wrote individual frames to folder {}".format(video_name, save_dir))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="A script to read the given video using PyAV and write its individual "
    #                                              "frames using PIL or skimage.")

    # parser.add_argument('-d', '--dir', help='Full path of directory where video is located', type=str, required=True)
    # parser.add_argument('-v', '--vid', help='Video name including the extension', type=str, required=True)
    # parser.add_argument('-w', '--writer', help='Which frame writer? 0: PIL; 1: skimage', type=int, choices=[0, 1],
    #                     required=True)
    # parser.add_argument('-c', '--crop_and_resize', help='crop and resize video frames while writing? (demonstrated with'
    #                                                     ' an example crop size)', action="store_true", default=False)
    # args = parser.parse_args()
    target='valid'
    root_dir=f"..\Breakfast Action Recognition\\{target}"
    
    if not os.path.exists(os.path.join('..','data',target)):
        os.mkdir(os.path.join('..','data',target))
    path=os.path.join('..','data',target)
    save_dir = path
    for person in os.listdir(root_dir):
        for video in os.listdir(os.path.join(root_dir,person)):
            video_frames_writer(os.path.join(root_dir,person), video, 0, False,save_dir)
    target='train'
    root_dir=f"..\Breakfast Action Recognition\\{target}"
    
    if not os.path.exists(os.path.join('..','data',target)):
        os.mkdir(os.path.join('..','data',target))
    path=os.path.join('..','data',target)
    save_dir = path
    for person in os.listdir(root_dir):
        for video in os.listdir(os.path.join(root_dir,person)):
            video_frames_writer(os.path.join(root_dir,person), video, 0, False,save_dir)