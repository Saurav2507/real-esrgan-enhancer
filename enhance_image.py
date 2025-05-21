import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from realesrgan import RealESRGAN

def enhance_image(input_path, output_path, user_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth')

    image = Image.open(input_path).convert('RGB')
    sr_image = model.predict(image)

    resized_image = sr_image.resize((1500, 1000))
    output_filename = f"{user_name}_enhanced.png"
    resized_image.save(os.path.join(output_path, output_filename))
    print(f"Enhanced image saved as {output_filename}")
