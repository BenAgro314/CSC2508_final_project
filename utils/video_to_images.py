import cv2
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def video_to_PIL(video_path, subsample_stride=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    for frame_number in range(0, total_frames, subsample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        yield frame_number, pil_image

    cap.release()