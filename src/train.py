# This software is provided "as is", without warranty of any kind, express or implied.
# It is intended for research and educational purposes only — not for medical or diagnostic use.
# See the LICENSE file for full terms and conditions.
import time 
import os 
from glob import glob 

import torch  
import torch.nn as nn
import torch.optim as o
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image 
from tqdm import tqdm   

import matplotlib.pyplot as plot 

class BinaryCancerDataset(Dataset): 
    def __init__(self, image_paths, labels, transform=None): 
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): 
        return len(self.image_paths)
    def __getitem__(self, sample_index): 
        path_to_image = self.image_paths[sample_index]
        image = Image.open(path_to_image)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        raw_label = self.labels[sample_index]
        label_tensor = torch.tensor(raw_label, dtype = torch.float32) 

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label_tensor

class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        self.net = nn.Sequential(  
            nn.Conv2d(3,16,3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Conv2d(16,32,3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Conv2d(32,64,3, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Conv2d(64,128,3, padding = 1),
            nn.ReLU(), 
            nn.MaxPool2d(2), 

            nn.Flatten(), 
            nn.Linear(128*14*14,256), 
            nn.ReLU(), 
            nn.Dropout(p=0.3), 
            nn.Linear(256,1), 
            nn.Sigmoid() 
    )
    def forward(self, data): 
        return self.net(data)

if __name__ == "__main__":
    print("Step 1: Checking for devices")
    if(torch.cuda.is_available()): 
        device = "cuda" 
    else:
        device = "cpu"
    print(f"Using {device}")

    print("Step 2: Image Loading")
    base_path  = r'' #Defines your base path, replace the space between the '' to input your own base path
    image_paths = glob(os.path.join(base_path, '**', '*.PNG'), recursive=True)  

    def BiLabel(folder_name):
        if '1' in folder_name: 
            return 1
        elif '0' in folder_name:
            return 0
        else:
            return None
    labels = []
    for p in image_paths:
        label = BiLabel(os.path.basename(os.path.dirname(p)))
        labels.append(label)
    count = 0
    while count < len(labels):
        if labels[count] == None:
            del image_paths[count]
            del labels[count]
        else:
            count += 1
    print(f"Total Images: {len(image_paths)}")
    print("Step 3: Normalizing data")

    class BinaryCancerDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, sample_index): 
            path_to_image = self.image_paths[sample_index]

            image = Image.open(path_to_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            raw_label = self.labels[sample_index]
            label_tensor = torch.tensor(raw_label, dtype = torch.float32) 

            if self.transform is not None:
                image = self.transform(image)
                
            return image, label_tensor

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = BinaryCancerDataset(image_paths, labels, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = 32,
        shuffle = True,
        num_workers = 2,
        pin_memory = True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size = 32,
        shuffle = False, 
        num_workers = 2,
        pin_memory = True
    )

    print("Step 4: Creating model")
    model = CNN().to(device)
   
    print("Step 5: Training Setup")
    criterion = nn.BCELoss()
    optimizer = o.Adam(model.parameters(), lr=0.001)
    epochs = 20

    train_accuracy = []
    train_loss = []
    print("Step 6: Training Starting")
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        loop = tqdm(train_loader, desc = f"Epoch {epoch+1}/{epochs}", leave=True)
        N=0
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1) 

            output = model(images)
            loss = criterion(output, labels) 

            optimizer.zero_grad()
            loss.backward()   
            optimizer.step()

            total_loss += loss.item()
            prediction = (output > 0.5).float()
            total_correct += (prediction == labels).sum().item()
            total_samples +=labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100*total_correct/total_samples)

            average_loss = total_loss/total_samples
            average_accuracy = total_correct/total_samples
            N+=1
            if (N==300):
                train_loss.append(average_loss)
                train_accuracy.append(average_accuracy)
                N=0
    elapsed = time.time() - start_time
    print(f"Epoch{epoch+1} completed in {elapsed:.2f}s | Accuracy: {100*total_correct/total_samples:.2f}%")
    torch.save(model.state_dict(),"model_weights.pth")
    print("Model saved for further use")
    
    

    print("Step 7: Test Data")
    model.eval()

    
    num_correct = 0
    num_total = 0
    
    start_time = time.time()
 
    with torch.no_grad(): 
        loop = tqdm(test_loader, desc= "Evaluating", leave=True)
        for test_images, test_labels in loop: 
            test_images = test_images.to(device) 
            test_labels = test_labels.to(device).unsqueeze(1)  
            
           
            predictions = model(test_images)
            predicted_labels = (predictions > 0.5).float()
            
            
            num_correct += (predicted_labels == test_labels).sum().item()
          
            num_total += test_labels.size(0)
            
 
    time_checker = time.time()-start_time
    test_accuracy = 100 * num_correct / num_total
    print(f"Test Accuracy: {test_accuracy:.2f}% | Completed in {time_checker:.2f}s")

    
    print("Step 8: Training Graphs")
    plot.figure()
    plot.plot(range(1, len(train_accuracy)+1), train_accuracy, marker='.', linestyle='-', alpha=0.6)
    plot.title("Accuracy per 300 batches")
    plot.xlabel("Batches (Scaled by 300)")
    plot.ylabel("Accuracy (%)")
    plot.grid(True)  
    plot.savefig("accuracy_by_batch.png")
    plot.show()
  
    plot.figure()
    plot.plot(range(1, len(train_accuracy)+1), train_loss, marker='.', linestyle='-', alpha=0.6)
    plot.title("Loss per 300 batches")
    plot.xlabel("Batches (Scaled by 300)")
    plot.ylabel("Loss")
    plot.grid(True) 
    plot.savefig("loss_by_batch.png")
    plot.show()






