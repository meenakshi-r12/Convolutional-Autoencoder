# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

In practical scenarios, images often contain noise that degrades the performance of computer vision models. A convolutional autoencoder learns compressed representations of images and reconstructs them, which can be used to remove noise.

Dataset: MNIST (28×28 grayscale images of handwritten digits)
Noise: Gaussian noise will be added to simulate real-world scenarios

## DESIGN STEPS

### Step 1: Setup Environment
Import required libraries: PyTorch, torchvision, matplotlib, and others for data handling and visualization.

### Step 2: Load Dataset
Download the MNIST dataset and apply transformations to convert images to tensors suitable for training.

### Step 3: Introduce Noise
Add Gaussian noise to the training and testing images using a custom noise-adding function.

### Step 4: Define Autoencoder Architecture
Encoder: Convolutional layers (Conv2D) with ReLU activations and MaxPooling
Decoder: Transposed convolutional layers (ConvTranspose2D) with ReLU and Sigmoid activations to reconstruct the image

### Step 5: Prepare Training
Initialize the autoencoder model
Define Mean Squared Error (MSE) as the loss function
Choose Adam optimizer for training

### Step 6: Model Training
Train the autoencoder using the noisy images as input and the original clean images as the target. Track the loss over epochs to monitor learning.

### Step 7: Evaluate and Visualize
Compare the original, noisy, and denoised images
Visualize results to assess the model’s performance in removing noise

## PROGRAM
### Name: MEENAKSHI R
### Register Number: 212224220062

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # 28x28 -> 14x14
            nn.Conv2d(16, 8, 3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)              # 14x14 -> 7x7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),    # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),    # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),             # 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Training Function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: VENKATANATHAN P R")
    print("Register Number: 212223240173")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

## OUTPUT

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
         MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4            [-1, 8, 14, 14]           1,160
              ReLU-5            [-1, 8, 14, 14]               0
         MaxPool2d-6              [-1, 8, 7, 7]               0
   ConvTranspose2d-7           [-1, 16, 14, 14]             528
              ReLU-8           [-1, 16, 14, 14]               0
   ConvTranspose2d-9            [-1, 8, 28, 28]             520
             ReLU-10            [-1, 8, 28, 28]               0
           Conv2d-11            [-1, 1, 28, 28]              73
          Sigmoid-12            [-1, 1, 28, 28]               0
================================================================
Total params: 2,441
Trainable params: 2,441
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.40
Params size (MB): 0.01
Estimated Total Size (MB): 0.41
----------------------------------------------------------------
```


### Original vs Noisy Vs Reconstructed Image
```
Epoch [1/5], Loss: 0.0195
Epoch [2/5], Loss: 0.0191
Epoch [3/5], Loss: 0.0187
Epoch [4/5], Loss: 0.0184
Epoch [5/5], Loss: 0.0181
Name: MEENAKSHI R                
Register Number:  212224220062                
```

<img width="1789" height="589" alt="download" src="https://github.com/user-attachments/assets/d76e39ce-aef6-44ea-9d2c-1d8602ce21ae" />

## RESULT

The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
