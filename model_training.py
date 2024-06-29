import segmentation_models_pytorch as smp
import torch.optim as optim

# Define the loss function
criterion = smp.losses.DiceLoss(mode='binary')
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.008)
import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)        
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

    
class UNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
    

# Get UNet model
model = UNet()

def train_model(model, dataloader, criterion, optimizer, num_epochs=500):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            # Zero the parameter gradients
            optimizer.zero_grad()
            print(f"Input shape: {inputs.shape}")
            
            print(f"Labels shape: {labels.shape}")
            # Forward pass
            outputs = model(inputs)['Out']
            print(f"Input shape: {inputs.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Labels shape: {labels.shape}")
            loss = criterion(outputs, labels)

 
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            print(epoch)
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    return model
# Train the model
trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=5)

torch.save(model.state_dict(), 'segmentation_model.pth')
model.load_state_dict(torch.load('segmentation_model.pth'))
model.eval()

def evaluate_model_on_test_data(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)['out']
            _, preds = torch.max(outputs, 1)

            # Visualize and/or save predictions
            for i in range(len(images)):
                image = images[i].cpu().numpy().transpose((1, 2, 0))  # Convert to numpy array
                prediction = preds[i].cpu().numpy()  # Convert to numpy array

                # Visualize or save the image and prediction
                visualize_prediction_or_save(image, prediction)  # Define your visualization or saving function

# Example usage
# Assuming you have a TestDataLoader named test_loader
evaluate_model_on_test_data(model, test_dataloader, device)

def visualize_prediction_or_save(image, prediction, save_path=None):
    # Create a subplot with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the predicted mask
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Optionally, save the visualization to disk
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Example usage
# Assuming 'image' and 'prediction' are numpy arrays representing the image and prediction respectively
visualize_prediction_or_save(image, prediction)"""
