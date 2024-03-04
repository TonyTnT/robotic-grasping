import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        print("Device Name: ", device.get_info(rs.camera_info.name))
        print("Device ID: ", device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

        print("Color Intrinsics: ", self.intrinsics)
        print("Depth Factor: ", self.scale)
        print("FINISH CONNECT")

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        
        if not aligned_depth_frame or not color_frame:
            # continue
            print("ERROR")

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth / self.scale, alpha=0.03), cv2.COLORMAP_JET)

        # 混合RGB图像和彩色深度图
        alpha = 0.5 # 可以调整这个值来改变混合图像的不透明度
        blended_image = cv2.addWeighted(rgb, alpha, depth_colormap, 1 - alpha, 0)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = rgb.shape
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(rgb, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap, blended_image))
        else:
            images = np.hstack((rgb, depth_colormap, blended_image))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=130322270731,width=1280,height=720,fps=30)
    cam.connect()
    while True:
        cam.plot_image_bundle()
        key = cv2.waitKey(1)
        if key==ord("q"):
            break
