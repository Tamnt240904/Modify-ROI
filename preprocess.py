import cv2
import numpy as np
import math

def create_shake_pattern(frame_count, intensity=5, frequency=0.5, fps=30):
    """
    Tạo pattern rung lắc tự nhiên liên tục (không còn dùng cho discrete shake)
    """
    time_array = np.arange(frame_count) / fps
    t = 2 * np.pi * frequency * time_array
    
    offset_x = (intensity * np.sin(2 * np.pi * t) + 
                intensity * 0.3 * np.sin(4 * np.pi * t + 0.5) +
                intensity * 0.2 * np.sin(7 * np.pi * t + 1.2))
    
    offset_y = (intensity * np.cos(2 * np.pi * t + 0.3) + 
                intensity * 0.3 * np.cos(4 * np.pi * t + 0.8) +
                intensity * 0.2 * np.cos(7 * np.pi * t + 2.0))
    
    return offset_x.astype(int), offset_y.astype(int)

def create_discrete_shake_pattern(frame_count, intensity=5, shake_interval=5.0, 
                                   shake_duration=0.5, fps=30, stable_start=0.0):
    """
    Tạo pattern rung lắc rời rạc - nhảy sang vị trí mới và giữ nguyên
    
    Args:
        frame_count: Số frame trong video
        intensity: Độ mạnh rung lắc (pixels)
        shake_interval: Khoảng thời gian giữa các lần lắc (giây)
        shake_duration: Thời gian chuyển động sang vị trí mới (giây)
        fps: Frame rate của video
        stable_start: Thời gian giữ nguyên ở đầu video (giây)
    
    Returns:
        offset_x, offset_y: Mảng các giá trị offset cho từng frame
    """
    offset_x = np.zeros(frame_count)
    offset_y = np.zeros(frame_count)
    
    frames_per_interval = int(shake_interval * fps)
    frames_per_shake = int(shake_duration * fps)
    stable_frames = int(stable_start * fps)
    
    # Giữ nguyên vị trí ban đầu cho stable_start giây đầu
    current_frame = stable_frames
    
    # Vị trí hiện tại (bắt đầu ở trung tâm)
    current_x = 0.0
    current_y = 0.0
    
    while current_frame < frame_count:
        # Tạo một vị trí ngẫu nhiên mới để nhảy tới
        target_x = np.random.uniform(-intensity, intensity)
        target_y = np.random.uniform(-intensity, intensity)
        
        shake_start = current_frame
        shake_end = min(shake_start + frames_per_shake, frame_count)
        
        # Chuyển động từ vị trí hiện tại sang vị trí mới với easing
        for i in range(shake_start, shake_end):
            if shake_start < frame_count:
                t = (i - shake_start) / frames_per_shake
                
                # Sử dụng ease-out để chuyển động tự nhiên
                # t_eased = 1 - (1 - t) ** 3  # Cubic ease-out
                t_eased = 1 - np.cos(t * np.pi / 2)  # Sine ease-out
                
                # Nội suy từ vị trí cũ sang vị trí mới
                offset_x[i] = current_x + (target_x - current_x) * t_eased
                offset_y[i] = current_y + (target_y - current_y) * t_eased
        
        # Giữ nguyên ở vị trí mới cho đến lần lắc tiếp theo
        next_shake = min(current_frame + frames_per_interval, frame_count)
        for i in range(shake_end, next_shake):
            offset_x[i] = target_x
            offset_y[i] = target_y
        
        # Cập nhật vị trí hiện tại
        current_x = target_x
        current_y = target_y
        
        current_frame += frames_per_interval
    
    return offset_x.astype(int), offset_y.astype(int)

def add_camera_shake(input_video, output_video, crop_percent=0.9, 
                     shake_intensity=10, shake_interval=5.0, 
                     shake_duration=0.5, discrete=True, stable_start=0.0):
    """
    Thêm hiệu ứng rung lắc camera vào video
    
    Args:
        input_video: Đường dẫn video đầu vào
        output_video: Đường dẫn video đầu ra
        crop_percent: Tỷ lệ vùng crop so với khung hình gốc (0.9 = 90%)
        shake_intensity: Độ mạnh rung lắc (pixels)
        shake_interval: Khoảng thời gian giữa các lần lắc (giây)
        shake_duration: Thời gian của mỗi cú lắc (giây)
        discrete: True = lắc rời rạc, False = lắc liên tục
        stable_start: Thời gian giữ nguyên ở đầu video (giây)
    """
    # Mở video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Không thể mở video!")
        return
    
    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tính kích thước vùng crop
    crop_width = int(frame_width * crop_percent)
    crop_height = int(frame_height * crop_percent)
    
    # Vị trí trung tâm ban đầu
    center_x = (frame_width - crop_width) // 2
    center_y = (frame_height - crop_height) // 2
    
    # Giới hạn di chuyển
    max_offset = min(center_x, center_y)
    
    print(f"Kích thước gốc: {frame_width}x{frame_height}")
    print(f"Kích thước crop: {crop_width}x{crop_height}")
    print(f"Tổng số frame: {total_frames}")
    print(f"Kiểu lắc: {'Rời rạc' if discrete else 'Liên tục'}")
    if stable_start > 0:
        print(f"Giữ nguyên {stable_start} giây đầu")
    print(f"Chu kỳ lắc: {shake_interval} giây")
    if discrete:
        print(f"Thời gian mỗi cú lắc: {shake_duration} giây")
    
    # Tạo pattern rung lắc
    if discrete:
        offset_x, offset_y = create_discrete_shake_pattern(
            total_frames, 
            min(shake_intensity, max_offset),
            shake_interval,
            shake_duration,
            fps,
            stable_start
        )
    else:
        shake_frequency = 1.0 / shake_interval
        offset_x, offset_y = create_shake_pattern(
            total_frames, 
            min(shake_intensity, max_offset),
            shake_frequency,
            fps
        )
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (crop_width, crop_height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tính vị trí crop với offset rung lắc
        x = center_x + offset_x[frame_idx]
        y = center_y + offset_y[frame_idx]
        
        # Đảm bảo không vượt biên
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))
        
        # Crop frame
        cropped = frame[y:y+crop_height, x:x+crop_width]
        
        # Ghi frame
        out.write(cropped)
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Đã xử lý: {frame_idx}/{total_frames} frames")
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Hoàn thành! Video đã được lưu tại: {output_video}")

def add_camera_shake_with_motion_blur(input_video, output_video, 
                                      crop_percent=0.9, 
                                      shake_intensity=10, 
                                      shake_interval=5.0,
                                      shake_duration=0.5,
                                      blur_amount=0.3,
                                      discrete=True,
                                      stable_start=0.0):
    """
    Thêm hiệu ứng rung lắc với motion blur để tự nhiên hơn
    
    Args:
        shake_interval: Khoảng thời gian giữa các lần lắc (giây)
        shake_duration: Thời gian của mỗi cú lắc (giây)
        discrete: True = lắc rời rạc, False = lắc liên tục
        stable_start: Thời gian giữ nguyên ở đầu video (giây)
    """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Không thể mở video!")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    crop_width = int(frame_width * crop_percent)
    crop_height = int(frame_height * crop_percent)
    
    center_x = (frame_width - crop_width) // 2
    center_y = (frame_height - crop_height) // 2
    max_offset = min(center_x, center_y)
    
    # Tạo pattern rung lắc
    if discrete:
        offset_x, offset_y = create_discrete_shake_pattern(
            total_frames, 
            min(shake_intensity, max_offset),
            shake_interval,
            shake_duration,
            fps,
            stable_start
        )
    else:
        shake_frequency = 1.0 / shake_interval
        offset_x, offset_y = create_shake_pattern(
            total_frames, 
            min(shake_intensity, max_offset),
            shake_frequency,
            fps
        )
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (crop_width, crop_height))
    
    prev_frame = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        x = center_x + offset_x[frame_idx]
        y = center_y + offset_y[frame_idx]
        
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))
        
        cropped = frame[y:y+crop_height, x:x+crop_width]
        
        # Thêm motion blur khi rung mạnh
        if prev_frame is not None and frame_idx > 0:
            # Tính độ di chuyển
            motion = abs(offset_x[frame_idx] - offset_x[frame_idx-1]) + \
                    abs(offset_y[frame_idx] - offset_y[frame_idx-1])
            
            if motion > 3:  # Nếu di chuyển nhanh
                # Blend với frame trước
                cropped = cv2.addWeighted(cropped, 1-blur_amount, 
                                         prev_frame, blur_amount, 0)
        
        out.write(cropped)
        prev_frame = cropped.copy()
        
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Đã xử lý: {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Hoàn thành! Video đã được lưu tại: {output_video}")

# Cách sử dụng:
if __name__ == "__main__":
    # Ví dụ 1: Giữ nguyên 10 giây đầu, sau đó lắc mỗi 5 giây
    add_camera_shake(
        input_video="data/input.mp4",
        output_video="data/output_shake.mp4",
        crop_percent=0.80,        # Crop 90% kích thước gốc
        shake_intensity=150,       # Độ xa có thể di chuyển (pixels)
        shake_interval=5.0,       # Nhảy sang vị trí mới mỗi 5 giây
        shake_duration=0.1,       # Thời gian chuyển động sang vị trí mới (0.3 giây)
        discrete=True,            # Lắc rời rạc
        stable_start=10.0         # Giữ nguyên 10 giây đầu
    )
    
    # Ví dụ 2: Giữ nguyên 5 giây đầu
    # add_camera_shake(
    #     input_video="input.mp4",
    #     output_video="output_stable5s.mp4",
    #     crop_percent=0.90,
    #     shake_intensity=20,
    #     shake_interval=4.0,
    #     shake_duration=0.2,
    #     discrete=True,
    #     stable_start=5.0         # Giữ nguyên 5 giây đầu
    # )
    
    # Ví dụ 3: Không giữ nguyên, lắc ngay từ đầu
    # add_camera_shake(
    #     input_video="input.mp4",
    #     output_video="output_immediate.mp4",
    #     crop_percent=0.90,
    #     shake_intensity=15,
    #     shake_interval=5.0,
    #     shake_duration=0.3,
    #     discrete=True,
    #     stable_start=0.0         # Lắc ngay từ đầu
    # )
    
    # Ví dụ 4: Giữ nguyên 10 giây với motion blur
    # add_camera_shake_with_motion_blur(
    #     input_video="input.mp4",
    #     output_video="output_shake_blur.mp4",
    #     crop_percent=0.90,
    #     shake_intensity=15,
    #     shake_interval=5.0,
    #     shake_duration=0.3,
    #     blur_amount=0.4,
    #     discrete=True,
    #     stable_start=10.0        # Giữ nguyên 10 giây đầu
    # )