import cv2
import numpy as np
import json

class ROITracker:
    def __init__(self, video_path, initial_roi):
        """
        Args:
            video_path: Đường dẫn video
            initial_roi: List of points [(x1,y1), (x2,y2), ...] định nghĩa ROI ban đầu
        """
        self.cap = cv2.VideoCapture(video_path)
        self.initial_roi = np.array(initial_roi, dtype=np.float32)
        self.current_roi = self.initial_roi.copy()
        
        # Feature detector - AKAZE tốt hơn cho rotation, fallback về ORB
        try:
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            print("Sử dụng AKAZE detector (tốt cho rotation)")
        except:
            self.detector = cv2.ORB_create(nfeatures=1500)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            print("Sử dụng ORB detector")
        
        # Lưu reference frame và features
        self.ref_frame = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        
        # Re-initialization parameters để tránh drift
        self.frame_count = 0
        self.reinit_interval = 30  # Re-init mỗi 30 frames
        self.last_good_roi = self.initial_roi.copy()
        
    def _get_background_mask(self, frame):
        """Tạo mask chỉ chứa background (loại bỏ foreground)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges (background thường có nhiều edges rõ ràng)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges để tạo vùng xung quanh edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Tính gradient magnitude (vùng có texture cao)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)
        
        # Threshold: vùng có gradient cao = background có texture
        _, texture_mask = cv2.threshold(gradient_mag, 30, 255, cv2.THRESH_BINARY)
        
        # Combine edges và texture
        bg_mask = cv2.bitwise_or(edges_dilated, texture_mask)
        
        # Morphological operations để clean
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_clean)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_clean)
        
        return bg_mask
    
    def _exclude_roi_mask(self, frame_shape, roi_points, margin=50):
        """Tạo mask loại bỏ vùng ROI + margin"""
        mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
        
        # Mở rộng ROI thêm margin
        roi_expanded = roi_points.copy()
        center = roi_expanded.mean(axis=0)
        roi_expanded = center + (roi_expanded - center) * (1 + margin/100)
        
        cv2.fillPoly(mask, [roi_expanded.astype(np.int32)], 0)
        
        return mask
    
    def initialize_reference(self, frame):
        """Khởi tạo reference frame và features"""
        self.ref_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tạo combined mask
        bg_mask = self._get_background_mask(frame)
        roi_mask = self._exclude_roi_mask(frame.shape, self.current_roi)
        combined_mask = cv2.bitwise_and(bg_mask, roi_mask)
        
        # Detect features chỉ trong vùng background ngoài ROI
        self.ref_keypoints, self.ref_descriptors = self.detector.detectAndCompute(
            gray, mask=combined_mask
        )
        
        print(f"Đã detect {len(self.ref_keypoints)} features từ background")
    
    def track_frame(self, frame):
        """Track camera shift và update ROI với HOMOGRAPHY"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Re-initialize reference mỗi N frames để tránh drift
        self.frame_count += 1
        if self.frame_count % self.reinit_interval == 0:
            print(f"\n=== Re-initializing reference at frame {self.frame_count} ===")
            self.initialize_reference(frame)
            return None
        
        # Tạo combined mask cho frame hiện tại
        bg_mask = self._get_background_mask(frame)
        roi_mask = self._exclude_roi_mask(frame.shape, self.current_roi)
        combined_mask = cv2.bitwise_and(bg_mask, roi_mask)
        
        # Detect features trong frame hiện tại
        curr_keypoints, curr_descriptors = self.detector.detectAndCompute(
            gray, mask=combined_mask
        )
        
        if curr_descriptors is None or len(curr_keypoints) < 10:
            print("Không đủ features để track!")
            return None
        
        # Match features
        matches = self.matcher.knnMatch(self.ref_descriptors, curr_descriptors, k=2)
        
        # Lọc good matches bằng Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            print(f"Chỉ có {len(good_matches)} good matches, không đủ để tính transformation!")
            return None
        
        # Lấy corresponding points
        ref_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches])
        
        # Tính HOMOGRAPHY matrix với RANSAC (xử lý perspective transformation + rotation)
        homography_matrix, inliers = cv2.findHomography(
            ref_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.995,
            maxIters=2000
        )
        
        if homography_matrix is None:
            print("Không tính được homography matrix!")
            return None
        
        # Kiểm tra quality của homography (tránh degenerate cases)
        num_inliers = np.sum(inliers)
        inlier_ratio = num_inliers / len(good_matches)
        
        if inlier_ratio < 0.25:  # Nếu quá ít inliers, không tin tưởng kết quả
            print(f"Inlier ratio quá thấp: {inlier_ratio:.2f}, bỏ qua frame này")
            return None
        
        # Kiểm tra homography có hợp lý không (không quá distorted)
        det = np.linalg.det(homography_matrix[:2, :2])
        if det < 0.1 or det > 10:  # Quá scale hoặc degenerate
            print(f"Homography không hợp lý (det={det:.3f}), bỏ qua frame này")
            return None
        
        # Áp dụng homography lên ROI (perspective transformation)
        roi_pts = self.initial_roi.reshape(-1, 1, 2).astype(np.float32)
        transformed_roi = cv2.perspectiveTransform(roi_pts, homography_matrix)
        self.current_roi = transformed_roi.reshape(-1, 2)
        
        # Lưu lại ROI tốt cho trường hợp cần fallback
        self.last_good_roi = self.current_roi.copy()
        
        print(f"Matches: {len(good_matches)}, Inliers: {num_inliers}, Ratio: {inlier_ratio:.2f}, Det: {det:.3f}")
        
        return {
            'transform_matrix': homography_matrix,
            'num_matches': len(good_matches),
            'num_inliers': num_inliers,
            'inlier_ratio': inlier_ratio,
            'ref_pts': ref_pts[inliers.ravel() == 1],
            'curr_pts': curr_pts[inliers.ravel() == 1],
            'bg_mask': bg_mask,
            'combined_mask': combined_mask
        }
    
    def draw_visualization(self, frame, track_result, show_initial=True):
        """Vẽ visualization với ROI ban đầu và ROI tracked"""
        vis = frame.copy()
        
        # Vẽ ROI ban đầu (màu đỏ, đường đứt nét)
        if show_initial:
            pts = self.initial_roi.astype(np.int32)
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % len(pts)])
                self._draw_dashed_line(vis, pt1, pt2, (0, 0, 255), thickness=3, gap=15)
        
        # Vẽ ROI hiện tại (màu xanh lá, đường liền nét)
        cv2.polylines(vis, [self.current_roi.astype(np.int32)], 
                     isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Tô màu semi-transparent cho ROI hiện tại
        overlay = vis.copy()
        cv2.fillPoly(overlay, [self.current_roi.astype(np.int32)], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        if track_result:
            # Vẽ matched points (background features)
            for pt in track_result['curr_pts']:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 7, (255, 255, 255), 2)
            
            # Vẽ vector di chuyển từ initial ROI đến tracked ROI
            init_center = self.initial_roi.mean(axis=0).astype(int)
            curr_center = self.current_roi.mean(axis=0).astype(int)
            cv2.arrowedLine(vis, tuple(init_center), tuple(curr_center), 
                           (0, 255, 255), 4, tipLength=0.2)
            
            # Tính displacement
            displacement = np.linalg.norm(curr_center - init_center)
            
            # Hiển thị thông tin với background
            info_text = [
                f"Frame: {self.frame_count}",
                f"Matches: {track_result['num_matches']}",
                f"Inliers: {track_result['num_inliers']}",
                f"Ratio: {track_result.get('inlier_ratio', 0):.2f}",
                f"Displacement: {displacement:.1f}px"
            ]
            
            for i, text in enumerate(info_text):
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(vis, (8, 20 + i*35), (text_width + 20, 50 + i*35), 
                             (0, 0, 0), -1)
                cv2.rectangle(vis, (8, 20 + i*35), (text_width + 20, 50 + i*35), 
                             (0, 255, 0), 2)
                cv2.putText(vis, text, (12, 45 + i*35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, gap=10):
        """Vẽ đường đứt nét"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
            pts.append((x, y))
        
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[i + 1], color, thickness)
    
    def process_video(self, output_path=None, show_masks=True):
        """Xử lý toàn bộ video"""
        # Đọc frame đầu tiên
        ret, frame = self.cap.read()
        if not ret:
            print("Không đọc được video!")
            return
        
        # Initialize reference
        self.initialize_reference(frame)
        
        # Setup video writer nếu cần
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            h, w = frame.shape[:2]
            
            if show_masks:
                out = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h*2))
            else:
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Tạo cửa sổ hiển thị
        win_masks = 'Tracking (Top-left: Result, Top-right: BG Mask, Bottom-left: Combined Mask, Bottom-right: Original)'
        win_vis = 'ROI Tracking'
        try:
            cv2.namedWindow(win_masks, cv2.WINDOW_NORMAL)
            cv2.namedWindow(win_vis, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win_masks, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(win_vis, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            print(f"\n--- Frame {self.frame_count} ---")
            
            # Track
            track_result = self.track_frame(frame)
            
            # Visualize
            vis = self.draw_visualization(frame, track_result)
            
            if show_masks and track_result:
                # Tạo grid 2x2: [vis, bg_mask, combined_mask, original]
                bg_mask_color = cv2.cvtColor(track_result['bg_mask'], cv2.COLOR_GRAY2BGR)
                combined_mask_color = cv2.cvtColor(track_result['combined_mask'], cv2.COLOR_GRAY2BGR)
                
                top_row = np.hstack([vis, bg_mask_color])
                bottom_row = np.hstack([combined_mask_color, frame])
                grid = np.vstack([top_row, bottom_row])
                
                cv2.imshow(win_masks, grid)
                
                if output_path:
                    out.write(grid)
            else:
                cv2.imshow(win_vis, vis)
                
                if output_path:
                    out.write(vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


# ===== DEMO USAGE =====
if __name__ == "__main__":
    # Thay đổi đường dẫn video của bạn ở đây
    video_path = "data/webcam.mp4"
    
    # Đọc video để lấy kích thước
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Không đọc được video!")
        exit()
    
    h, w = first_frame.shape[:2]
    cap.release()
    
    print(f"Kích thước video: {w}x{h}")
    
    # Cố gắng đọc ROI từ file JSON
    json_path = "data/webcam.json"
    normalized_roi = None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            pts = data.get('points') if isinstance(data, dict) else None
            if pts and isinstance(pts, list):
                parsed = []
                for p in pts:
                    if isinstance(p, dict) and 'x' in p and 'y' in p:
                        parsed.append((float(p['x']), float(p['y'])))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        parsed.append((float(p[0]), float(p[1])))
                if len(parsed) >= 3:
                    normalized_roi = parsed
    except Exception as e:
        print(f"Không đọc được JSON ROI từ {json_path}: {e}")

    # Fallback nếu không có dữ liệu hợp lệ
    if normalized_roi is None:
        normalized_roi = [
            (0.27, 0.62), (0.47, 0.62), (0.56, 0.77), (0.3, 0.78)
        ]
        print("Dùng ROI mặc định (không tìm thấy/không hợp lệ file JSON).")
    else:
        print(f"Đã load ROI từ {json_path}: {normalized_roi}")

    # Chuyển normalized coords sang pixel
    max_coord = max(max(x, y) for (x, y) in normalized_roi)
    if max_coord <= 1.5:
        initial_roi = [(int(x * w), int(y * h)) for (x, y) in normalized_roi]
    else:
        initial_roi = [(int(x), int(y)) for (x, y) in normalized_roi]
    
    print(f"ROI tại vị trí: {initial_roi}")
    
    # Tạo tracker với Homography support
    tracker = ROITracker(video_path, initial_roi)
    
    # Xử lý video
    tracker.process_video(
        output_path="output/output_tracked.mp4",
        show_masks=True
    )
    
    print("\nHoàn tất!")