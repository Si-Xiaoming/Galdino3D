import torch
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from scipy.ndimage import map_coordinates
import numpy as np
import os
from video_utils import extract_frames
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

class D3VideoGenerator:
    def __init__(self, video_path, work_space):
        self.video_path = video_path
        self.work_space = work_space

    def generate_depth_maps(self, frames_dir, output_dir):
        '''生成深度图
        Args:
            frames_dir: 帧目录
            output_dir: 输出目录
        '''
        os.makedirs(output_dir, exist_ok=True)

        # 加载 MiDaS 模型
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()
        transform = Compose([
            Resize((384,384)),
            #ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for frame_name in sorted(os.listdir(frames_dir)):
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)
            # 转为浮点数
            frame = frame.astype("float32") / 255.0
            # [H, W, C] -> [C, H, W]
            frame = frame.transpose(2, 0, 1)
            frame = torch.from_numpy(frame)
            
            input_tensor = transform(frame).unsqueeze(0)


            # 深度推理
            with torch.no_grad():
                depth_map = model(input_tensor).squeeze().cpu().numpy()

            # 保存深度图
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
            cv2.imwrite(os.path.join(output_dir, frame_name), depth_map.astype("uint8"))

    def estimate_optical_flow(self, prev_frame, next_frame):
        '''估计光流
        Args:
            prev_frame: 前一帧
            next_frame: 后一帧
        Returns:
            flow: 光流
        '''
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) # 光流估计
        return flow

    def align_depth_with_flow(self, depth_map, flow):
        '''使用光流对齐深度图
        Args:
            depth_map: 深度图
            flow: 光流
        Returns:
            aligned_depth: 对齐后的深度图
        '''
        h, w = depth_map.shape
        flow_x, flow_y = flow[..., 0], flow[..., 1]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        new_x = np.clip(grid_x + flow_x, 0, w - 1)
        new_y = np.clip(grid_y + flow_y, 0, h - 1)
        aligned_depth = map_coordinates(depth_map, [new_y, new_x], order=1)
        return aligned_depth

    def optimize_depth_consistency(self, frames_dir, depth_maps_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        frame_files = sorted(os.listdir(frames_dir))
        depth_files = sorted(os.listdir(depth_maps_dir))

        for i in range(len(frame_files) - 1):
            # 加载帧和深度图
            frame = cv2.imread(os.path.join(frames_dir, frame_files[i]))
            next_frame = cv2.imread(os.path.join(frames_dir, frame_files[i + 1]))
            depth_map = cv2.imread(os.path.join(depth_maps_dir, depth_files[i]), cv2.IMREAD_GRAYSCALE)

            # 估计光流并对齐深度
            flow = self.estimate_optical_flow(frame, next_frame)
            aligned_depth = self.align_depth_with_flow(depth_map, flow)

            # 保存优化后的深度图
            cv2.imwrite(os.path.join(output_dir, depth_files[i]), aligned_depth)

    def generate_stereo_views(self, frames_dir, depth_maps_dir, left_output_dir, right_output_dir, baseline=0.1):
        '''生成左右视图
        Args:
            frames_dir: 帧目录
            depth_maps_dir: 深度图目录
            left_output_dir: 左视图输出目录
            right_output_dir: 右视图输出目录
            baseline: 基线距离
        '''
        os.makedirs(left_output_dir, exist_ok=True)
        os.makedirs(right_output_dir, exist_ok=True)

        for frame_name in sorted(os.listdir(frames_dir)):
            frame = cv2.imread(os.path.join(frames_dir, frame_name))
            depth_map = cv2.imread(os.path.join(depth_maps_dir, frame_name), cv2.IMREAD_GRAYSCALE)

            if depth_map is not None:
                h, w = depth_map.shape
                disparity = baseline / (depth_map.astype(np.float32) + 1e-6)

                left_view = np.zeros_like(frame)
                right_view = np.zeros_like(frame)

                for y in range(h):
                    for x in range(w):
                        offset = int(disparity[y, x])
                        if x - offset >= 0:
                            left_view[y, x] = frame[y, x - offset]
                        if x + offset < w:
                            right_view[y, x] = frame[y, x + offset]

                cv2.imwrite(os.path.join(left_output_dir, frame_name), left_view)
                cv2.imwrite(os.path.join(right_output_dir, frame_name), right_view)


    def refine_stereo_images(self, input_dir, output_dir, prompt="Enhance stereo view"):
        '''修复视图 (使用 stable-diffusion 模型)
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            prompt: RunwayML 模型提示
        '''
        os.makedirs(output_dir, exist_ok=True)
        # pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to("cuda")
        pretrain_file = r'.\Cache\huggingface'
        pipe = StableDiffusionInpaintPipeline.from_pretrained(pretrain_file, local_files_only=True).to("cuda")
        pipe.enable_attention_slicing()

        for image_name in sorted(os.listdir(input_dir)):
            image_path = os.path.join(input_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            result = pipe(prompt=prompt, image=image, height=256, width=256).images[0]
            result.save(os.path.join(output_dir, image_name))


    def create_anaglyph_video(self, left_dir, right_dir, output_path):
        '''创建立体视频 (红蓝眼)
        Args:
            left_dir: 左视图目录
            right_dir: 右视图目录
            output_path: 输出路径
        '''
        left_files = sorted(os.listdir(left_dir))
        right_files = sorted(os.listdir(right_dir))

        h, w, _ = cv2.imread(os.path.join(left_dir, left_files[0])).shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        for left_name, right_name in zip(left_files, right_files):
            left_image = cv2.imread(os.path.join(left_dir, left_name))
            right_image = cv2.imread(os.path.join(right_dir, right_name))

            anaglyph = np.zeros_like(left_image)
            anaglyph[:, :, 2] = left_image[:, :, 2]  # 红通道
            anaglyph[:, :, :2] = right_image[:, :, :2]  # 绿蓝通道

            out.write(anaglyph)

        out.release()



    def create_side_by_side_video(self, left_dir, right_dir, output_path):
        '''创建立体视频 (左右视图)
        Args:
            left_dir: 左视图目录
            right_dir: 右视图目录
            output_path: 输出路径
        '''
        left_files = sorted(os.listdir(left_dir))
        right_files = sorted(os.listdir(right_dir))

        h, w, _ = cv2.imread(os.path.join(left_dir, left_files[0])).shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 30, (w * 2, h))

        for left_name, right_name in zip(left_files, right_files):
            left_image = cv2.imread(os.path.join(left_dir, left_name))
            right_image = cv2.imread(os.path.join(right_dir, right_name))

            combined = np.hstack((left_image, right_image))
            out.write(combined)

        out.release()


    def execuate(self):
        # 在work_space下创建文件夹frames, depth_maps, aligned_depth_maps, left_views, right_views, refined_left_views, refined_right_views
        frames_dir = os.path.join(self.work_space, "frames")
        depth_maps_dir = os.path.join(self.work_space, "depth_maps")
        aligned_depth_maps_dir = os.path.join(self.work_space, "aligned_depth_maps")
        left_views_dir = os.path.join(self.work_space, "left_views")
        right_views_dir = os.path.join(self.work_space, "right_views")
        refined_left_views_dir = os.path.join(self.work_space, "refined_left_views")
        refined_right_views_dir = os.path.join(self.work_space, "refined_right_views")

        # 分帧
        #extract_frames(self.video_path, frames_dir)
        # 1. 生成深度图
        #self.generate_depth_maps(frames_dir, depth_maps_dir)

        # 2. 优化深度一致性
        #self.optimize_depth_consistency(frames_dir, depth_maps_dir, aligned_depth_maps_dir)

        # 3. 生成左右视图
        #self.generate_stereo_views(frames_dir, aligned_depth_maps_dir, left_views_dir, right_views_dir)

        # 4. 修复左右视图
        self.refine_stereo_images(left_views_dir, refined_left_views_dir)
        self.refine_stereo_images(right_views_dir, refined_right_views_dir)

        # 5. 创建立体视频
        self.create_anaglyph_video(refined_left_views_dir, refined_right_views_dir, "anaglyph.mp4")
        self.create_side_by_side_video(refined_left_views_dir, refined_right_views_dir, "side_by_side.mp4")

def main():
    # 设置缓存路径
    os.environ['TORCH_HOME'] = r'D:\03-codeLearning\python\Galdino3D\Cache'

    # 设置代理
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'

    video_path = r'D:\03-codeLearning\python\Galdino3D\WorkSpace\input.mp4'
    work_space = r'D:\03-codeLearning\python\Galdino3D\WorkSpace'
    generator = D3VideoGenerator(video_path, work_space)
    generator.execuate()

if __name__ == "__main__":
    main()