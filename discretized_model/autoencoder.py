import numpy as np
import torch
from torch import nn
import torch.nn.functional as func
from torchvision import transforms
from matplotlib import pyplot as plt, cm
from IPython import display


def to_image(img):
    # img has values between -1 and 1 after tanh
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    return img


def normalize_image(img):
    img = img / 255
    img = transforms.Normalize((0.5,), (0.5,))(img)
    return img


class Autoencoder(nn.Module):
    def __init__(self, hidden_size=32, image_size=56, batch_size=1024, dtype=None):
        super().__init__()

        if dtype is None:
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        else:
            self.dtype = dtype

        self.loss = []
        self.number_examples = 3

        # sizes
        self.len_h = hidden_size
        self.image_size = image_size
        self.batch_size = batch_size

        # NOTE: sizes are always [batch_size, channels, width, height] !!!
        # ENCODER
        self.enc_conv1 = nn.Conv2d(1, self.len_h // 2, kernel_size=4, stride=2, padding=1)  # b, len_h//2, 28, 28
        self.enc_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # b, len_h//2, 14, 14
        self.enc_dropout = nn.Dropout2d(p=0.1)
        self.enc_conv2 = nn.Conv2d(self.len_h // 2, self.len_h, kernel_size=4, stride=2, padding=1)  # b, len_h, 7, 7
        self.enc_conv2_bn = nn.BatchNorm2d(self.len_h)
        self.enc_maxpool2 = nn.MaxPool2d(kernel_size=self.image_size // 8, stride=1, padding=0)  # b, len_h, 1, 1
        # reshape from [b, len_h, 1, 1] to  [b, len_h]

        # DECODER
        # reshape from [b, len_h] to [b, len_h, 1, 1]
        self.dec_upconv1 = nn.ConvTranspose2d(self.len_h, 64, kernel_size=7, stride=1, padding=0)  # b, 64, 7, 7
        self.dec_upconv1_bn = nn.BatchNorm2d(64)
        self.dec_upconv2 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)  # b, 16, 14, 14
        self.dec_upconv2_bn = nn.BatchNorm2d(16)
        self.dec_upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)  # b, 8, 28, 28
        self.dec_upconv3_bn = nn.BatchNorm2d(8)
        self.dec_upconv4 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)  # b, 1, 56, 56

    def encoder(self, img):
        h = func.relu(self.enc_conv1(img))
        h = self.enc_maxpool1(h)
        h = self.enc_dropout(h)
        h = func.relu(self.enc_conv2_bn(self.enc_conv2(h)))
        h = self.enc_maxpool2(h)
        h = torch.reshape(h, [h.size(0), h.size(1)])
        return h

    def decoder(self, h):
        x = torch.reshape(h, [h.size(0), h.size(1), 1, 1])
        x = func.relu(self.dec_upconv1_bn(self.dec_upconv1(x)))
        x = func.relu(self.dec_upconv2_bn(self.dec_upconv2(x)))
        x = func.relu(self.dec_upconv3_bn(self.dec_upconv3(x)))
        x = torch.tanh(self.dec_upconv4(x))
        return x

    def forward(self, img):
        h = self.encoder(img)
        pred_img = self.decoder(h)
        return pred_img

    def train_net(self, dataloader, max_epochs=10000, sample_length=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        dashboard = TrainingDashboard(self.number_examples,
                                      max_epochs=max_epochs, sample_length=sample_length)
        dashboard.set_epoch_length(epoch_length=max_epochs)

        print('Training using type:', self.dtype)

        for run in range(max_epochs // sample_length):
            dataloader.load_data(perc_normal=0.1, perc_distractors=0.0)
            self.training_run(dataloader, dashboard, optimizer, max_epochs=sample_length)

        dashboard.update_plot()
        plt.savefig('../images/autoencoder_training.png', transparent=False)

        return self.loss

    def training_run(self, dataloader, dashboard, optimizer, max_epochs):
        criterion = nn.MSELoss()

        for epoch in range(max_epochs):
            # run through one epoch
            for _, img, _, _ in dataloader:
                img = normalize_image(img)
                # Forward pass
                pred_img = self(img)
                loss = criterion(img, pred_img)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.loss = np.append(self.loss, [loss.item()])

            # log, make plot
            if (epoch + 1) % 10 == 0 or epoch == 0:
                indices = np.random.randint(len(img), size=self.number_examples)
                # img_ = to_image(img[indices]).cpu().data
                # pred_img_ = to_image(pred_img[indices]).cpu().data

                img_ = img[indices].cpu().data
                pred_img_ = pred_img[indices].cpu().data

                dashboard.set_loss(loss=self.loss)
                dashboard.set_data(images=img_, pred_images=pred_img_)
                dashboard.update_plot()


class TrainingDashboard:
    def __init__(self, number_examples=3, max_epochs=10000, sample_length=100):
        # data
        self.images = None
        self.pred_images = None

        self.loss = None

        self.max_epochs = max_epochs
        self.sample_length = sample_length

        # plot
        self.number_examples = number_examples
        self.columns = 2

        self.fig = plt.figure(constrained_layout=True, figsize=(self.columns * 3.2, self.number_examples * 4.2))
        self.axs = np.full(shape=(self.number_examples, self.columns), fill_value=None)
        gs = self.fig.add_gridspec(self.number_examples + 1, self.columns)

        for i in range(self.number_examples):
            for j in range(self.columns):
                self.axs[i, j] = self.fig.add_subplot(gs[i, j], adjustable='box')

        self.loss_axs = self.fig.add_subplot(gs[self.number_examples, :])

        self.fig.suptitle('Training Progress: Digit Autoencoder', fontsize=24)

    def set_data(self, images, pred_images):
        self.images = images
        self.pred_images = pred_images

    def set_loss(self, loss):
        self.loss = loss

    def set_epoch_length(self, epoch_length=100):
        self.max_epochs = epoch_length

    def update_plot(self):
        assert len(self.images) == len(self.pred_images) \
               >= self.number_examples, 'Data elements do not have the same shape or are too short!'

        for pos in range(len(self.images)):
            for j in range(self.columns):
                self.axs[pos, j].clear()

            self.axs[pos, 0].imshow(flatten_channel_dim(self.images)[pos], cmap=cm.gray_r)
            self.axs[pos, 0].set_xticks([])
            self.axs[pos, 0].set_yticks([])

            self.axs[pos, 1].imshow(flatten_channel_dim(self.pred_images)[pos], cmap=cm.gray_r)
            self.axs[pos, 1].set_xticks([])
            self.axs[pos, 1].set_yticks([])

        self.axs[self.number_examples - 1, 1].set_xlabel('Digit')

        self.axs[0, 0].set_title('original image')
        self.axs[0, 1].set_title('predicted image')

        if self.loss is not None:
            self.loss_axs.clear()
            self.loss_axs.plot(range(len(self.loss)), self.loss, zorder=1)
            self.loss_axs.set_title(f'epoch {len(self.loss)}/{self.max_epochs}')

            lines = range(0, self.max_epochs, self.sample_length)
            self.loss_axs.vlines(lines, min(self.loss), max(self.loss), zorder=0, color='lightgrey', alpha=0.5)

        self.loss_axs.set_xlim(0, self.max_epochs)
        self.loss_axs.set_xlabel('Epoch')
        self.loss_axs.set_ylabel('Loss')

        display.display(plt.gcf())
        display.clear_output(wait=True)


# util functions
def flatten_channel_dim(imgs):
    imgs = imgs.cpu()
    return torch.reshape(imgs, [imgs.shape[0], imgs.shape[2], imgs.shape[3]])
