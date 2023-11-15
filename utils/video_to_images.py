import cv2
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_video_into_images(video_path, img_size, device, subsample_stride=1):
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

        transform = transforms.Compose([
            transforms.Resize((img_size,img_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        image = transform(pil_image).unsqueeze(0).to(device)
        yield frame_number, image, pil_image
    cap.release()

def video_to_PIL(video_path, subsample_stride=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    pils = []

    for frame_number in range(0, total_frames, subsample_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pils.append(pil_image)

    return pils