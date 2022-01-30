import warnings
import torch
from torch import nn
import torch.nn.functional as func
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import display


# util functions
def flatten_channel_dim(imgs):
    imgs = imgs.cpu()
    return torch.reshape(imgs, [imgs.shape[0], imgs.shape[2], imgs.shape[3]])


def to_image(img):
    # img has values between -1 and 1 after tanh
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    return img


def normalize_image(img):
    img = img / 255
    img = transforms.Normalize((0.5,), (0.5,))(img)
    return img


class Decoder(nn.Module):
    def __init__(self, classes=10, states=2, hidden_size=32, image_size=56, dtype=None, batch_size=1024):
        super().__init__()

        if dtype is None:
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        else:
            self.dtype = dtype

        self.number_examples = 3
        self.loss = []

        # sizes
        self.len_z = states
        self.len_c = classes
        self.len_h = hidden_size
        self.image_size = image_size
        self.batch_size = batch_size

        # NOTE: sizes are always [batch_size, channels, width, height] !!!

        # ENCODER
        # ENCODER
        self.enc_conv1 = nn.Conv2d(1, self.len_h // 2, kernel_size=4, stride=2, padding=1)  # b, len_h//2, 28, 28
        self.enc_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # b, len_h//2, 14, 14
        self.enc_dropout = nn.Dropout2d(p=0.1)
        self.enc_conv2 = nn.Conv2d(self.len_h // 2, self.len_h, kernel_size=4, stride=2, padding=1)  # b, len_h, 7, 7
        self.enc_conv2_bn = nn.BatchNorm2d(self.len_h)
        self.enc_maxpool2 = nn.MaxPool2d(kernel_size=self.image_size // 8, stride=1, padding=0)  # b, len_h, 1, 1
        # reshape from [b, len_h, 1, 1] to  [b, len_h]

        # DECODER
        # concat: h and z
        # reshape: [b, h+z] => [b, h+z, 1, 1]
        self.upconv1 = nn.ConvTranspose2d(self.len_h + self.len_z, 64, kernel_size=7, stride=1,
                                          padding=0)  # b, 64, 7, 7
        self.upconv1_bn = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # b, 32, 14, 14
        self.upconv2_bn = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # b, 16, 28, 28
        self.upconv3_bn = nn.BatchNorm2d(16)
        self.upconv4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)  # b, 1, 56, 56

        # CLASSIFIER
        self.cls_fc1 = nn.Linear(self.len_h, 64)
        self.cls_dropout = nn.Dropout(p=0.4)
        self.cls_fc2 = nn.Linear(64, self.len_c)
        # softmax

    def encoder(self, img):
        h = func.relu(self.enc_conv1(img))
        h = self.enc_maxpool1(h)
        h = self.enc_dropout(h)
        h = func.relu(self.enc_conv2_bn(self.enc_conv2(h)))
        h = self.enc_maxpool2(h)
        h = torch.reshape(h, [h.size(0), h.size(1)])
        return h

    def decoder(self, states):
        assert states.shape[-1] == self.len_h + self.len_z, 'States do not have right shape!'

        x = torch.reshape(states, [states.shape[0], self.len_h + self.len_z, 1, 1])

        x = func.relu(self.upconv1_bn(self.upconv1(x)))
        x = func.relu(self.upconv2_bn(self.upconv2(x)))
        x = func.relu(self.upconv3_bn(self.upconv3(x)))
        x = torch.tanh(self.upconv4(x))
        return x

    def classifier(self, hidden):
        c = func.relu(self.cls_fc1(hidden))
        c = self.cls_dropout(c)
        c = torch.softmax(self.cls_fc2(c), dim=1)
        return c

    def train_net(self, dataloader, max_epochs=10000, sample_length=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

        dashboard = TrainingDashboard(self.number_examples,
                                      max_epochs=max_epochs, sample_length=sample_length)
        dashboard.set_epoch_length(epoch_length=max_epochs)

        print('Training using type:', self.dtype)

        for run in range(max_epochs // sample_length):
            dataloader.load_data(perc_normal=0.1)
            self.train_decoder(dataloader, dashboard, optimizer, max_epochs=sample_length)

        dashboard.update_plot()
        plt.savefig('../images/big_decoder_training.jpg', transparent=False)

        return self.loss

    def train_decoder(self, dataloader, dashboard, optimizer, max_epochs):
        criterion = CustomLoss(l_img=20, l_cls=1)
        for epoch in range(max_epochs):
            # run through one epoch
            for or_img, trns_img, cls, pose in dataloader:
                trns_img = normalize_image(trns_img)
                or_img = normalize_image(or_img)
                # Forward pass
                hidden = self.encoder(or_img)
                states = torch.concat((hidden, pose), dim=1)
                pred_img = self.decoder(states)
                pred_cls = self.classifier(hidden)

                loss = criterion(trns_img, pred_img, cls, pred_cls)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.loss = np.append(self.loss, [loss.item()])

            # log, make plot
            if (epoch + 1) % 10 == 0 or epoch == 0:
                indices = np.random.randint(len(pred_img), size=self.number_examples)
                pred_img_ = pred_img[indices].cpu().data
                pred_cls_ = pred_cls[indices].cpu().data
                img_ = trns_img[indices].cpu().data
                pose_ = pose[indices].cpu().data
                cls_ = cls[indices].cpu().data

                dashboard.set_loss(loss=self.loss)
                dashboard.set_data(images=img_, pred_images=pred_img_,
                                   true_cls=cls_, pred_cls=pred_cls_,
                                   pose=pose_)
                dashboard.update_plot()


class CustomLoss(nn.Module):
    def __init__(self, l_img=1, l_cls=1):
        self.l_img = l_img
        self.l_latent = l_cls
        super(CustomLoss, self).__init__()

    def forward(self, img_inputs, img_outputs, cls_inputs, cls_outputs):
        image_loss = nn.MSELoss()
        feature_loss = nn.KLDivLoss()

        img_inputs = img_inputs.view(-1)
        img_outputs = img_outputs.view(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = self.l_img * image_loss(img_inputs, img_outputs) + self.l_latent * feature_loss(cls_inputs,
                                                                                                   cls_outputs)
        return loss


class TrainingDashboard:
    def __init__(self, number_examples=3, max_epochs=10000, sample_length=100):
        # data
        self.images = None
        self.pred_images = None
        self.true_cls = None
        self.pred_cls = None
        self.pose = None

        self.loss = None

        self.max_epochs = max_epochs
        self.sample_length = sample_length

        # plot
        self.number_examples = number_examples

        self.fig = plt.figure(constrained_layout=True, figsize=(10, self.number_examples * 4.2))
        self.axs = np.full(shape=(self.number_examples, 4), fill_value=None)
        gs = self.fig.add_gridspec(self.number_examples + 1, 4)

        for i in range(self.number_examples):
            for j in range(4):
                self.axs[i, j] = self.fig.add_subplot(gs[i, j])

        self.loss_axs = self.fig.add_subplot(gs[self.number_examples, :])

        self.fig.suptitle('Training Progress: Big Decoder', fontsize=24)

    def set_data(self, images, pred_images, true_cls, pred_cls, pose):
        self.images = images
        self.pred_images = pred_images
        self.true_cls = true_cls
        self.pred_cls = pred_cls
        self.pose = pose

    def set_loss(self, loss):
        self.loss = loss

    def set_epoch_length(self, epoch_length=100):
        self.max_epochs = epoch_length

    def update_plot(self):
        assert len(self.images) \
               == len(self.true_cls) \
               == len(self.pred_cls) \
               == len(self.pose) \
               >= self.number_examples, 'Data elements do not have the same shape or are too short!'

        for pos in range(len(self.images)):
            for j in range(4):
                self.axs[pos, j].clear()

            self.axs[pos, 0].imshow(flatten_channel_dim(self.images)[pos], cmap=cm.gray_r)
            self.axs[pos, 0].set_xticks([])
            self.axs[pos, 0].set_yticks([])

            self.axs[pos, 1].bar(range(self.pred_cls.shape[1]), self.pred_cls[pos], color='red', zorder=1)
            self.axs[pos, 1].bar(range(self.true_cls.shape[1]), self.true_cls[pos], color='lightgray', zorder=0)
            self.axs[pos, 1].set_xticks(range(10))
            self.axs[pos, 1].set_ylabel('Probability')
            self.axs[pos, 1].set_ylim(0, 1)

            self.axs[pos, 2].vlines(range(2), -4, 4, colors='grey', alpha=0.7, zorder=0)
            self.axs[pos, 2].hlines(0, -1, 2, colors='grey', alpha=0.7, zorder=0)
            self.axs[pos, 2].scatter(range(self.pose.shape[1]), self.pose[pos], c='green', zorder=1)
            self.axs[pos, 2].set_ylim(-3.5, 3.5)
            self.axs[pos, 2].set_xlim(-0.5, 1.5)
            self.axs[pos, 2].set_xticks([])

            if self.pred_images is not None:
                self.axs[pos, 3].imshow(flatten_channel_dim(self.pred_images)[pos], cmap=cm.gray_r)
            self.axs[pos, 3].set_xticks([])
            self.axs[pos, 3].set_yticks([])

        self.axs[self.number_examples - 1, 1].set_xlabel('Digit')
        self.axs[self.number_examples - 1, 2].set_xticks(range(2))
        self.axs[self.number_examples - 1, 2].set_xticklabels(['scale', 'rotation'], rotation=60, fontsize=12)

        self.axs[0, 0].set_title('original image')
        self.axs[0, 1].set_title('class')
        self.axs[0, 2].set_title('pose')
        self.axs[0, 3].set_title('predicted image')

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
