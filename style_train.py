from models.style import *
import torch
import numpy as np
import os
import sys
import random
from PIL import Image
import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class StyleTrainer():
    def __init__(self, style_image='path/to/style.jpg', style_name='style_name', dataset_path='path/to/dataset/',
                 epochs=8,batch_size=8,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_name = style_name
        self.style_image = style_image
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.output_dir = f"./style-images/outputs/{style_name}-training"
        self.checkpoint_dir = f"./checkpoints/"
        self.image_size = 256
        self.style_size = 448
        self.batch_size = 8
        self.lr = 1e-5
        self.epochs = 8
        self.checkpoint_model = None
        self.checkpoint_interval = 500
        self.sample_interval = 500
        self.lambda_style = 10e10
        self.lambda_content = 10e5
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        """ Dataset and dataloader """
        self.train_dataset = datasets.ImageFolder(self.dataset_path, self.train_transform(self.image_size))
        self.dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)

        """ Define networks """
        self.transformer = TransformerNet().to(self.device)
        self.vgg = VGG16(requires_grad=False).to(self.device)

        """ Define optimizer and loss """
        self.optimizer = Adam(self.transformer.parameters(), self.lr)
        self.MSEloss = torch.nn.MSELoss().to(self.device)

        """ Load style image """
        self.style = self.style_transform(self.style_size)(Image.open(style_image))
        self.style = self.style.repeat(self.batch_size, 1, 1, 1).to(self.device)

        """ Extract style features """
        self.features_style = self.vgg(self.style)
        self.gram_style = [self.gram_matrix(y) for y in self.features_style]

        """ Sample 8 images for visual evaluation of the model """
        self.image_samples = []
        for path in random.sample(glob.glob(f"{dataset_path}/*/*.jpg"), 8):
            self.image_samples += [self.style_transform(self.image_size)(Image.open(path))]
        self.image_samples = torch.stack(self.image_samples)

    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        # reshape the image as a matrix of pixel features
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        # transpose the matrix for the multiplication
        features_t = features.transpose(1, 2)
        # multiply the transposed matrix by the original matrix
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def train_transform(self, image_size):
        """ Transforms for training images """
        width = image_size
        height = image_size
        transform = transforms.Compose(
            [
                transforms.Resize((int(width * 1.15), int(height * 1.15))),
                transforms.RandomCrop((width, height)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        return transform

    def style_transform(self, image_size=None):
        """ Transforms for style image """

        width = image_size
        height = image_size

        resize = [transforms.Resize((height, width))] if image_size else []
        transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform

    def test_transform(self, image_size=None):
        """ Transforms for test image """
        width = image_size
        height = image_size

        resize = [transforms.Resize((height, width))] if image_size else []
        transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform

    def denormalize(self, tensors):
        """ Denormalizes image tensors using mean and std """

        for i in range(tensors.shape[0]):
            # direct inplace assignment cause problem in newer version of pytorch (lightning maybe?)
            for j in range(3):
                tensors[i, j, :] = tensors[i, j, :] * self.std[j] + self.mean[j]
        return tensors
    # use CMRFNN to denoise the image

    def deprocess(self, image_tensor):
        """ Denormalizes and rescales image tensor """
        image_tensor = self.denormalize(image_tensor)[0] # Normalizes the image tensor with mean and standard deviation of the dataset
        image_tensor *= 255 # Multiplies the image tensor by 255
        image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8) # Clips the image tensor to range [0, 255] and converts it to a numpy array
        image_np = image_np.transpose(1, 2, 0) # Transposes the image tensor to (H, W, C)
        return image_np

    def save_sample(self, batches_done):
        """ Evaluates the model and saves image samples """
        self.transformer.eval()
        with torch.no_grad():
            output = self.transformer(self.image_samples.to(self.device))
        image_grid = self.denormalize(torch.cat((self.image_samples.cpu(), output.cpu()), 2))
        save_image(image_grid, f"./style-images/outputs/{self.style_name}-training/{batches_done}.jpg", nrow=4)
        self.transformer.train()

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        """ Load checkpoint model if specified """
        if self.checkpoint_model:
            self.transformer.load_state_dict(torch.load(self.checkpoint_model))

        train_metrics = {"content": [], "style": [], "total": []}
        for epoch in range(self.epochs):
            epoch_metrics = {"content": [], "style": [], "total": []}
            for batch_i, (images, _) in enumerate(self.dataloader):
                self.optimizer.zero_grad()

                images_original = images.to(self.device)
                images_transformed = self.transformer(images_original)

                # Extract features
                features_original = self.vgg(images_original)
                features_transformed = self.vgg(images_transformed)

                # Compute content loss as MSE between features
                content_loss = self.lambda_content * self.MSEloss(features_transformed.relu2_2,
                                                                  features_original.relu2_2)

                # Compute style loss as MSE between gram matrices
                style_loss = 0
                for ft_y, gm_s in zip(features_transformed, self.gram_style):
                    gm_y = self.gram_matrix(ft_y)
                    style_loss += self.MSEloss(gm_y, gm_s[: images.size(0), :, :])
                style_loss *= self.lambda_style

                total_loss = content_loss + style_loss
                total_loss.backward()
                self.optimizer.step()

                epoch_metrics["content"] += [content_loss.item()]
                epoch_metrics["style"] += [style_loss.item()]
                epoch_metrics["total"] += [total_loss.item()]

                train_metrics["content"] += [content_loss.item()]
                train_metrics["style"] += [style_loss.item()]
                train_metrics["total"] += [total_loss.item()]

                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"
                    % (
                        epoch + 1,
                        self.epochs,
                        batch_i,
                        len(self.train_dataset),
                        content_loss.item(),
                        np.mean(epoch_metrics["content"]),
                        style_loss.item(),
                        np.mean(epoch_metrics["style"]),
                        total_loss.item(),
                        np.mean(epoch_metrics["total"]),
                    )
                )

                batches_done = epoch * len(self.dataloader) + batch_i + 1
                if batches_done % self.sample_interval == 0:
                    self.save_sample(batches_done)

                if self.checkpoint_interval > 0 and batches_done % self.checkpoint_interval == 0:
                    torch.save(self.transformer.state_dict(), f"./checkpoints/{self.style_name}_{batches_done}.pth")

                torch.save(self.transformer.state_dict(), f"./checkpoints/last_checkpoint.pth")

        print("Training Completed!")

        # printing the loss curve.
        plt.plot(train_metrics["content"], label="Content Loss")
        plt.plot(train_metrics["style"], label="Style Loss")
        plt.plot(train_metrics["total"], label="Total Loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()


# main function
if __name__ == '__main__':
    system = StyleTrainer(style_image='./data/styles/post_impressionism.jpg', style_name='post_impressionism',
                          dataset_path='./data/style_data/', epochs=8)
    system.train()
    # save sample image after style transfer
    image_path = 'path/to/image.jpg'
    checkpoint_model = './checkpoints/last_checkpoint.pth'
    save_path = './'
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)

    transform = system.test_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(system.device)
    transformer.load_state_dict(torch.load(checkpoint_model))
    transformer.eval()
    import cv2
    from torch.autograd import Variable
    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(system.device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = system.denormalize(transformer(image_tensor)).cpu()
    # Save image
    fn = checkpoint_model.split('/')[-1].split('.')[0]
    save_image(stylized_image, os.path.join(save_path, f"results/{fn}-output.jpg"))
    print("Image Saved!")
    plt.imshow(cv2.cvtColor(cv2.imread(os.path.join(save_path, f"results/{fn}-output.jpg")), cv2.COLOR_BGR2RGB))
