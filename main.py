import cv2
import numpy as np
import json
import sys
from collections import deque

class ROIStabilizer:
    def __init__(self, detector_type='ORB', smoothing_window=5, 
                 homography_confidence=0.99, outlier_threshold=3.0):
        """
        Khởi tạo ROI Stabilizer với các tham số chống rung
        detector_type: 'ORB', 'SIFT', hoặc 'AKAZE'
        smoothing_window: Số frame để làm mượt (temporal smoothing)
        homography_confidence: Độ tin cậy RANSAC (0.95-0.999)
        outlier_threshold: Ngưỡng loại bỏ outlier (pixel)
        """
        self.detector_type = detector_type
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.roi_points = None
        
        # Tham số chống rung
        self.smoothing_window = smoothing_window
        self.homography_history = deque(maxlen=smoothing_window)
        self.roi_history = deque(maxlen=smoothing_window)
        self.homography_confidence = homography_confidence
        self.outlier_threshold = outlier_threshold
        
        # Khởi tạo feature detector
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=3000)  # Tăng số features
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def set_reference_roi(self, frame, roi_points):
        """
        Đặt frame tham chiếu và ROI ban đầu
        """
        self.reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.roi_points = roi_points
        
        # Detect features trên toàn bộ frame
        self.reference_keypoints, self.reference_descriptors = \
            self.detector.detectAndCompute(self.reference_frame, None)
        
        # Clear history khi set reference mới
        self.homography_history.clear()
        self.roi_history.clear()
        
        print(f"Đã phát hiện {len(self.reference_keypoints)} keypoints trong reference frame")
    
    def smooth_homography(self, H):
        """
        Làm mượt homography matrix bằng cách lấy trung bình
        """
        if H is None:
            return None
        
        self.homography_history.append(H)
        
        if len(self.homography_history) < 2:
            return H
        
        # Tính trung bình có trọng số (frame gần đây có trọng số cao hơn)
        weights = np.linspace(0.5, 1.0, len(self.homography_history))
        weights = weights / weights.sum()
        
        smoothed_H = np.zeros((3, 3))
        for i, h in enumerate(self.homography_history):
            smoothed_H += weights[i] * h
        
        return smoothed_H
    
    def smooth_roi_points(self, new_points):
        """
        Làm mượt tọa độ ROI theo thời gian
        """
        self.roi_history.append(new_points)
        
        if len(self.roi_history) < 2:
            return new_points
        
        # Exponential moving average
        alpha = 0.3  # Hệ số làm mượt (0-1), càng nhỏ càng mượt
        
        smoothed_points = []
        for i in range(len(new_points)):
            x_smooth = new_points[i][0]
            y_smooth = new_points[i][1]
            
            # Lấy trung bình từ history
            for j, hist_points in enumerate(self.roi_history):
                if j < len(self.roi_history) - 1:
                    weight = (1 - alpha) ** (len(self.roi_history) - 1 - j)
                    x_smooth = alpha * x_smooth + (1 - alpha) * hist_points[i][0]
                    y_smooth = alpha * y_smooth + (1 - alpha) * hist_points[i][1]
            
            smoothed_points.append([x_smooth, y_smooth])
        
        return smoothed_points
    
    def filter_matches_by_distance(self, matches, src_pts, dst_pts):
        """
        Lọc matches dựa trên khoảng cách geometric
        """
        if len(matches) < 10:
            return matches, src_pts, dst_pts
        
        # Tính khoảng cách di chuyển của mỗi match
        distances = np.linalg.norm(dst_pts.reshape(-1, 2) - src_pts.reshape(-1, 2), axis=1)
        
        # Loại bỏ outliers bằng median absolute deviation
        median_dist = np.median(distances)
        mad = np.median(np.abs(distances - median_dist))
        threshold = median_dist + self.outlier_threshold * mad
        
        # Lọc matches
        mask = distances < threshold
        filtered_matches = [m for i, m in enumerate(matches) if mask[i]]
        filtered_src = src_pts[mask]
        filtered_dst = dst_pts[mask]
        
        return filtered_matches, filtered_src, filtered_dst
    
    def stabilize_roi(self, current_frame):
        """
        Tính toán vị trí ROI mới với chống rung
        """
        if self.reference_frame is None:
            return None
        
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features trong frame hiện tại
        current_kp, current_desc = self.detector.detectAndCompute(gray, None)
        
        if current_desc is None or len(current_kp) < 10:
            # Sử dụng ROI từ history nếu có
            if len(self.roi_history) > 0:
                return self.roi_history[-1], 0
            return self.roi_points, 0
        
        # Match features
        matches = self.matcher.knnMatch(self.reference_descriptors, current_desc, k=2)
        
        # Lọc matches tốt bằng Lowe's ratio test (nghiêm ngặt hơn)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Giảm từ 0.75 xuống 0.7
                    good_matches.append(m)
        
        if len(good_matches) < 15:  # Tăng threshold từ 10 lên 15
            if len(self.roi_history) > 0:
                return self.roi_history[-1], len(good_matches)
            return self.roi_points, len(good_matches)
        
        # Lấy tọa độ các điểm tương ứng
        src_pts = np.float32([self.reference_keypoints[m.queryIdx].pt 
                              for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([current_kp[m.trainIdx].pt 
                              for m in good_matches]).reshape(-1, 1, 2)
        
        # Lọc matches dựa trên khoảng cách
        good_matches, src_pts, dst_pts = self.filter_matches_by_distance(
            good_matches, src_pts, dst_pts
        )
        
        if len(good_matches) < 15:
            if len(self.roi_history) > 0:
                return self.roi_history[-1], len(good_matches)
            return self.roi_points, len(good_matches)
        
        # Tính homography với RANSAC nghiêm ngặt
        H, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=3.0,  # Giảm từ 5.0 xuống 3.0
            confidence=self.homography_confidence
        )
        
        if H is None:
            if len(self.roi_history) > 0:
                return self.roi_history[-1], len(good_matches)
            return self.roi_points, len(good_matches)
        
        # Làm mượt homography
        H_smoothed = self.smooth_homography(H)
        
        # Transform các điểm ROI
        h, w = current_frame.shape[:2]
        roi_corners = np.float32([[p[0] * w, p[1] * h] for p in self.roi_points]).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(roi_corners, H_smoothed)
        
        # Normalize lại về 0-1
        new_points = []
        for corner in transformed_corners:
            x_norm = np.clip(corner[0][0] / w, 0, 1)
            y_norm = np.clip(corner[0][1] / h, 0, 1)
            new_points.append([x_norm, y_norm])
        
        # Làm mượt tọa độ ROI
        smoothed_points = self.smooth_roi_points(new_points)
        
        # Đếm số inliers
        inliers = np.sum(mask) if mask is not None else 0
        
        return smoothed_points, inliers


def load_roi_from_json(json_path):
    """
    Đọc ROI từ file JSON
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        points = data.get('points', [])
        roi_points = [(p['x'], p['y']) for p in points]
        
        if len(roi_points) < 3:
            raise ValueError("ROI phải có ít nhất 3 điểm")
        
        print(f"Đã load ROI với {len(roi_points)} điểm từ {json_path}")
        return roi_points
    
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        return None


def draw_roi_polygon(frame, roi_points, color=(0, 255, 0), thickness=2):
    """
    Vẽ ROI polygon lên frame
    """
    h, w = frame.shape[:2]
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi_points], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return pts


def get_roi_mask(frame_shape, roi_points):
    """
    Tạo mask cho vùng ROI
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi_points], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_roi_region(frame, roi_points):
    """
    Trích xuất vùng ROI từ frame
    """
    mask = get_roi_mask(frame.shape, roi_points)
    
    # Tìm bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    x, y, w, h = cv2.boundingRect(contours[0])
    
    if w <= 0 or h <= 0:
        return None
    
    # Crop và apply mask
    roi_region = frame[y:y+h, x:x+w].copy()
    mask_crop = mask[y:y+h, x:x+w]
    
    roi_region[mask_crop == 0] = 0
    
    return roi_region


def main():
    # Đường dẫn file
    video_path = 'data/webcam.mp4'
    json_path = 'data/webcam.json'
    
    # Kiểm tra arguments từ command line
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    if len(sys.argv) >= 3:
        json_path = sys.argv[2]
    
    print(f"Video: {video_path}")
    print(f"ROI JSON: {json_path}")
    
    # Load ROI từ JSON
    roi_points = load_roi_from_json(json_path)
    if roi_points is None:
        print("Không thể load ROI. Thoát chương trình.")
        return
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return
    
    # Đọc frame đầu tiên
    ret, first_frame = cap.read()
    if not ret:
        print("Không thể đọc frame")
        return
    
    print(f"Video size: {first_frame.shape[1]}x{first_frame.shape[0]}")
    
    # Cấu hình kích thước cửa sổ hiển thị cố định
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    
    # Tạo cửa sổ với kích thước cố định
    window_name = 'ROI Stabilization (Anti-Shake)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Hiển thị ROI ban đầu
    display_frame = first_frame.copy()
    draw_roi_polygon(display_frame, roi_points, color=(0, 255, 0), thickness=2)
    display_resized = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow(window_name, display_resized)
    cv2.waitKey(2000)
    
    # Khởi tạo stabilizer với tham số chống rung
    stabilizer = ROIStabilizer(
        detector_type='ORB',
        smoothing_window=7,  # Tăng window size để mượt hơn
        homography_confidence=0.995,
        outlier_threshold=2.5
    )
    stabilizer.set_reference_roi(first_frame, roi_points)
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('stabilized_output.mp4', fourcc, fps,
                          (first_frame.shape[1], first_frame.shape[0]))
    
    frame_count = 0
    
    # Reset video về đầu
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\nBắt đầu xử lý video...")
    print("Nhấn 'q' để thoát")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Tính toán vị trí ROI mới
        new_roi_points, inliers_count = stabilizer.stabilize_roi(frame)
        
        # Vẽ ROI mới
        color = (0, 255, 0) if inliers_count >= 15 else (0, 165, 255)
        draw_roi_polygon(frame, new_roi_points, color=color, thickness=2)
        
        # Hiển thị thông tin
        cv2.putText(frame, f"Inliers: {inliers_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Smoothing: ON", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Trích xuất và hiển thị vùng ROI
        roi_region = extract_roi_region(frame, new_roi_points)
        if roi_region is not None and roi_region.size > 0:
            try:
                roi_resized = cv2.resize(roi_region, (200, 150))
                h, w = frame.shape[:2]
                frame[10:160, w-210:w-10] = roi_resized
            except:
                pass
        
        # Ghi frame
        out.write(frame)
        
        # Hiển thị
        cv2.imshow('ROI Stabilization (Anti-Shake)', frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Đã xử lý {frame_count} frames")
    print(f"✓ FPS: {fps}")
    print("✓ Video đã được lưu: stabilized_output.mp4")


if __name__ == "__main__":
    main()