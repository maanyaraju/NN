pip install torch torchvision opencv-python numpy

mport cv2
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import transforms

# Load the pre-trained ESRGAN model
class ESRGAN(torch.nn.Module):
    def __init__(self):
        super(ESRGAN, self).__init__()
        self.model = torch.load("ESRGAN.pth")
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESRGAN().to(device)

# Function to enhance number plates
def enhance_number_plate(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Super-resolution enhancement
    enhanced_tensor = model(image_tensor).clamp(0, 1)
    enhanced_image = to_pil_image(enhanced_tensor.squeeze().cpu())

    # Save enhanced image
    enhanced_image.save(output_path)
    print(f"Enhanced image saved at {output_path}")

# Example usage
input_image = "number_plate.jpg"  # Path to the input blurry number plate image
output_image = "enhanced_plate.jpg"  # Path to save the enhanced image

enhance_number_plate(input_image, output_image)
