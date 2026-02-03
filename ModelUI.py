# This software is provided "as is", without warranty of any kind, express or implied.
# It is intended for research and educational purposes only — not for medical or diagnostic use.
# See the LICENSE file for full terms and conditions.
import torch 
from training import CNN 
from torchvision import transforms 
import gradio as gr 
from PIL import Image 
model = CNN() 
model.load_state_dict(torch.load("model_weights.pth", map_location="cuda")) 
model.eval() 
if(torch.cuda.is_available()): 
    device = "cuda" 
else:
    device = "cpu"  
model.to(device) 
def prediction(image): 
    transform = transforms.Compose([  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    Normalized_image = transform(image).unsqueeze(0).to(device) 
    output = model(Normalized_image) 
    return "Cancer Detected" if output == 1 else "No cancer detected" 

UserInterface = gr.Interface(
    inputs = gr.Image(type = "pil"), 
    fn = prediction, 
    outputs = "text",
    title = "Binary Identification of Breast Cancer",
    description = "Please upload a histology image to detect if cancer is present. ⚠️ This tool is for research/educational purposes only. Not for medical use."
)

UserInterface.launch(share=True) 
    
