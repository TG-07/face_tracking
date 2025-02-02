import cv2

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cap.get(cv2.CAP_PROP_FPS))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height

video_path = "/Users/tanisha/Documents/Carnegie Mellon/intern/face_tracking/data/2/video.mp4"  # Replace with your video file path
resolution = get_video_resolution(video_path)
if resolution:
    print(f"Video Resolution: {resolution[0]}x{resolution[1]}")