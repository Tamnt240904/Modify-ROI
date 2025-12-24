import cv2
import numpy as np

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
        
        # Feature detector (ORB - nhanh và miễn phí)
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Lưu reference frame và features
        self.ref_frame = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        
    def _get_background_mask(self, frame):
        """Tạo mask chỉ chứa background (loại bỏ foreground)"""
        # GIẢI PHÁP 1: Không dùng background subtractor, chỉ dùng edge detection
        # để tìm vùng có cấu trúc rõ ràng (đường, vạch kẻ, rào chắn)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges (background thường có nhiều edges rõ ràng)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges để tạo vùng xung quanh edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Vùng có nhiều edges = vùng background tốt để track
        # Nhưng ta vẫn muốn loại bỏ xe → dùng simple thresholding
        # Xe thường có màu đồng nhất, background (đường, cỏ) có texture
        
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
        """Track camera shift và update ROI"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
        
        # Tính transformation matrix với RANSAC
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            ref_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )
        
        if transform_matrix is None:
            print("Không tính được transformation matrix!")
            return None
        
        # Áp dụng transformation lên ROI
        roi_homogeneous = np.hstack([self.initial_roi, np.ones((len(self.initial_roi), 1))])
        transformed_roi = roi_homogeneous @ transform_matrix.T
        self.current_roi = transformed_roi
        
        num_inliers = np.sum(inliers)
        print(f"Matches: {len(good_matches)}, Inliers: {num_inliers}")
        
        return {
            'transform_matrix': transform_matrix,
            'num_matches': len(good_matches),
            'num_inliers': num_inliers,
            'ref_pts': ref_pts[inliers.ravel() == 1],
            'curr_pts': curr_pts[inliers.ravel() == 1],
            'bg_mask': bg_mask,
            'combined_mask': combined_mask
        }
    
    def draw_visualization(self, frame, track_result, show_initial=True):
        """Vẽ visualization với ROI ban đầu và ROI tracked"""
        vis = frame.copy()
        
        # Vẽ ROI ban đầu (màu đỏ, đường đứt nét, RẤT DÀY)
        if show_initial:
            pts = self.initial_roi.astype(np.int32)
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % len(pts)])
                # Vẽ đường đứt nét DÀY
                self._draw_dashed_line(vis, pt1, pt2, (0, 0, 255), thickness=3, gap=15)
            

        
        # Vẽ ROI hiện tại (màu xanh lá, đường liền nét, RẤT DÀY)
        cv2.polylines(vis, [self.current_roi.astype(np.int32)], 
                     isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Tô màu semi-transparent cho ROI hiện tại (SÁNG HƠN)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [self.current_roi.astype(np.int32)], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        # Label cho ROI hiện tại (CHỮ TO, CÓ BACKGROUND)
        center_current = self.current_roi.mean(axis=0).astype(int)
        text = "TRACKED ROI"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        # Background trắng cho text
        # cv2.rectangle(vis, (center_current[0] - text_w//2 - 10, center_current[1] + 10),
        #              (center_current[0] + text_w//2 + 10, center_current[1] + text_h + 20), 
        #              (255, 255, 255), -1)
        # cv2.rectangle(vis, (center_current[0] - text_w//2 - 10, center_current[1] + 10),
        #              (center_current[0] + text_w//2 + 10, center_current[1] + text_h + 20), 
        #              (0, 255, 0), 3)
        # cv2.putText(vis, text, (center_current[0] - text_w//2, center_current[1] + text_h + 15),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        if track_result:
            # Vẽ matched points (background features) - TO HƠN
            for pt in track_result['curr_pts']:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 7, (255, 255, 255), 2)
            
            # Vẽ vector di chuyển từ initial ROI đến tracked ROI - DÀY HƠN
            init_center = self.initial_roi.mean(axis=0).astype(int)
            curr_center = self.current_roi.mean(axis=0).astype(int)
            cv2.arrowedLine(vis, tuple(init_center), tuple(curr_center), 
                           (0, 255, 255), 4, tipLength=0.2)
            
            # Tính displacement
            displacement = np.linalg.norm(curr_center - init_center)
            
            # Hiển thị thông tin - CHỮ TO HƠN
            # info_text = [
            #     f"Matches: {track_result['num_matches']}",
            #     f"Inliers: {track_result['num_inliers']}",
            #     f"Displacement: {displacement:.1f}px"
            # ]
            
            # Background cho text
            # for i, text in enumerate(info_text):
            #     (text_width, text_height), _ = cv2.getTextSize(
            #         text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            #     )
            #     cv2.rectangle(vis, (8, 20 + i*40), (text_width + 20, 55 + i*40), 
            #                  (0, 0, 0), -1)
            #     cv2.rectangle(vis, (8, 20 + i*40), (text_width + 20, 55 + i*40), 
            #                  (0, 255, 0), 2)
            #     cv2.putText(vis, text, (12, 48 + i*40), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Legend - TO HƠN
        legend_y = vis.shape[0] - 100
        # cv2.rectangle(vis, (10, legend_y - 5), (280, vis.shape[0] - 10), (0, 0, 0), -1)
        # cv2.rectangle(vis, (10, legend_y - 5), (280, vis.shape[0] - 10), (255, 255, 255), 2)
        # cv2.putText(vis, "Legend:", (20, legend_y + 20), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Đường đỏ đứt nét
        # self._draw_dashed_line(vis, (20, legend_y + 40), (60, legend_y + 40), (0, 0, 255), 4, 8)
        # cv2.putText(vis, "Initial ROI", (70, legend_y + 45), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Đường xanh liền
        # cv2.line(vis, (20, legend_y + 60), (60, legend_y + 60), (0, 255, 0), 4)
        # cv2.putText(vis, "Tracked ROI", (70, legend_y + 65), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Chấm xanh dương
        # cv2.circle(vis, (40, legend_y + 80), 5, (255, 0, 0), -1)
        # cv2.circle(vis, (40, legend_y + 80), 7, (255, 255, 255), 2)
        # cv2.putText(vis, "BG Features", (70, legend_y + 85), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            
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
                
                cv2.imshow('Tracking (Top-left: Result, Top-right: BG Mask, Bottom-left: Combined Mask, Bottom-right: Original)', grid)
                
                if output_path:
                    out.write(grid)
            else:
                cv2.imshow('ROI Tracking', vis)
                
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
    video_path = "data/output_shake.mp4"
    
    # Đọc video để lấy kích thước
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Không đọc được video!")
        exit()
    
    h, w = first_frame.shape[:2]
    cap.release()
    
    print(f"Kích thước video: {w}x{h}")
    
    # TỰ ĐỘNG TẠO ROI Ở GIỮA FRAME (20% kích thước frame)
    # Bạn có thể điều chỉnh các tọa độ này cho phù hợp
    roi_width = int(w * 0.2)
    roi_height = int(h * 0.2)
    roi_center_x = w // 2
    roi_center_y = h // 2
    # print(h, w)
    # breakpoint()
    normalized_roi = [
        [0.27, 0.62], [0.47, 0.62], [0.56, 0.77], [0.3, 0.78]
    ]
    initial_roi = [
        (int(x * w), int(y * h)) for (x, y) in normalized_roi
    ]
    
    print(f"ROI tại vị trí: {initial_roi}")
    print("Nếu muốn đổi vị trí ROI, hãy sửa các tọa độ trong initial_roi")
    
    # Tạo tracker
    tracker = ROITracker(video_path, initial_roi)
    
    # Xử lý video
    tracker.process_video(
        output_path="output/output_tracked.mp4",  # Set None nếu không muốn save
        show_masks=True  # Hiển thị background mask và combined mask
    )
    
    print("\nHoàn tất!")

