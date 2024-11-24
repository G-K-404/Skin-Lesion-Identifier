import torch
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import knime.scripting.io as knio
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 8
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)
model.train()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_table = knio.input_tables[0].to_pandas()
image_column = 'Path' 
label_column = 'trueLabel'

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx][image_column]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = int(self.dataframe.iloc[idx][label_column])
        
        return image, label

dataset = ImageDataset(dataframe=input_table, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device).long()
        
        
        optimizer.zero_grad()
        
       
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")


torch.save(model.state_dict(), r"models\densenet121_trained_model.pth")


output_df = pd.DataFrame({})
knio.output_tables[0] = knio.Table.from_pandas(output_df) 