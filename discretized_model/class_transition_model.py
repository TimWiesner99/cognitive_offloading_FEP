import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as func
from torchvision import transforms
from matplotlib import pyplot as plt, cm
from IPython import display


def normalize_image(img):
    img = img / 255
    img = transforms.Normalize((0.5,), (0.5,))(img)
    return img


def to_image(img):
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    return img


class TransitionModel(nn.Module):
    def __init__(self, transformation_states=2, classes=10, image_size=56, dtype=None):
        super().__init__()

        if dtype is None:
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        else:
            self.dtype = dtype

        self.loss = []
        self.number_examples = 3

        # sizes
        self.len_trns = transformation_states
        self.len_cls = classes
        self.image_size = image_size

        # IMAGE TRANSFORMER
        self.image_transformer = ImageTransformer(image_size=self.image_size, dtype=self.dtype)

        # CLASSIFIER
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)  # b, 8, 28, 28
        self.conv1_bn = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # b, 8, 14, 14
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)  # b, 16, 7, 7
        self.conv2_bn = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # b, 16, 3, 3
        self.conv_dropout = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # b, 32, 1, 1
        # reshape here: [b, 32, 1, 1 ] -> [b, 32]
        self.fc1 = nn.Linear(32, 10)
        # softmax here

    def classifier(self, image):
        try:
            image = torch.reshape(image, [len(image), 1, self.image_size, self.image_size])
        except:
            print('Unable to reshape. Image does not have right size!')

        x = func.relu(self.conv1_bn(self.conv1(image)))
        x = self.maxpool1(x)
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.conv_dropout(x)
        x = func.relu(self.conv3(x))
        x = torch.reshape(x, [len(x), 32])
        x = func.softmax(self.fc1(x), dim=1)

        return x

    def forward(self, img, action):
        scale, rotation = action
        trns_img = self.image_transformer(img, scale, rotation)
        new_cls = self.classifier(trns_img)

        return new_cls

    def train_classifier(self, dataloader, max_epochs=1000, sample_length=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        dashboard = TrainingDashboard(number_examples=self.number_examples,
                                      max_epochs=max_epochs,
                                      sample_length=sample_length)
        dashboard.set_epoch_length(epoch_length=max_epochs)

        print('Training using type:', self.dtype)

        for run in range(max_epochs // sample_length):
            dataloader.load_data(perc_normal=0.25, perc_distractors=0.25)
            self.training_run(dataloader, dashboard, optimizer, max_epochs=sample_length)

        dashboard.update_plot()
        plt.savefig('../images/class_transition_model_classifier_training.png', transparent=False)

        return self.loss

    def training_run(self, dataloader, dashboard, optimizer, max_epochs):
        criterion = nn.KLDivLoss(reduction='batchmean')

        for epoch in range(max_epochs):
            # run through one epoch
            for _, img, cls, _ in dataloader:
                img = normalize_image(img)
                # Forward pass
                pred_cls = self.classifier(img)
                loss = criterion(cls, pred_cls)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.loss = np.append(self.loss, [loss.item()])

            # log, make plot
            if (epoch + 1) % 10 == 0 or epoch == 0:
                indices = np.random.randint(len(cls), size=self.number_examples)
                img_ = img[indices].cpu().data
                cls_ = cls[indices].cpu().data
                pred_cls_ = pred_cls[indices].cpu().data

                dashboard.set_loss(loss=self.loss)
                dashboard.set_data(img=img_, cls=cls_, pred_cls=pred_cls_)
                dashboard.update_plot()


class ImageTransformer(nn.Module):
    def __init__(self, image_size=56, dtype=None):
        super().__init__()

        if dtype is None:
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        else:
            self.dtype = dtype

        self.image_size = image_size

    def forward(self, image, scale, rotation):
        if type(scale) is not torch.Tensor:
            scale = [torch.Tensor([scale])]
        if type(rotation) is not torch.Tensor:
            rotation = [torch.Tensor([rotation])]

        matrices = torch.stack([self.transformation_matrix(s, r) for (s, r) in zip(scale, rotation)]).type(self.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = func.affine_grid(matrices, image.size()).type(self.dtype)
            trns_image = func.grid_sample(image, grid, padding_mode='border').type(self.dtype)

        return trns_image

    def transformation_matrix(self, scale=1, rotation=0):
        # scale
        T_scale = torch.Tensor([[scale, 0, 0],
                                [0, scale, 0],
                                [0, 0, 1]]).type(self.dtype)
        # rotation
        T_rot = torch.Tensor([[torch.cos(rotation), -torch.sin(rotation), 0],
                              [torch.sin(rotation), torch.cos(rotation), 0],
                              [0, 0, 1]]).type(self.dtype)

        # combine matrices
        T = T_rot @ T_scale
        T_inv = torch.linalg.inv(T).type(self.dtype)
        return T_inv[0:2, :]


class TrainingDashboard:
    def __init__(self, number_examples=3, max_epochs=10000, sample_length=100):
        # data
        self.images = None
        self.true_cls = None
        self.pred_cls = None

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

        self.fig.suptitle('Training Progress: Transition Model Classifier', fontsize=24)

    def set_data(self, img, cls, pred_cls):
        self.images = img
        self.true_cls = cls
        self.pred_cls = pred_cls

    def set_loss(self, loss):
        self.loss = loss

    def set_epoch_length(self, epoch_length=100):
        self.max_epochs = epoch_length

    def update_plot(self):
        assert len(self.images) == len(self.true_cls) == len(self.pred_cls) \
               >= self.number_examples, 'Data elements do not have the same shape or are too short!'

        for pos in range(len(self.images)):
            for j in range(self.columns):
                self.axs[pos, j].clear()

            self.axs[pos, 0].imshow(flatten_channel_dim(self.images)[pos], cmap=cm.gray_r)
            self.axs[pos, 0].set_xticks([])
            self.axs[pos, 0].set_yticks([])

            self.axs[pos, 1].bar(range(self.pred_cls.shape[1]), self.pred_cls[pos], color='red', zorder=1)
            self.axs[pos, 1].bar(range(self.true_cls.shape[1]), self.true_cls[pos], color='lightgray', zorder=0)
            self.axs[pos, 1].set_xticks(range(10))
            self.axs[pos, 1].set_ylabel('Probability')
            self.axs[pos, 1].set_ylim(0, 1)

        self.axs[self.number_examples - 1, 1].set_xlabel('Digit')

        self.axs[0, 0].set_title('image')
        self.axs[0, 1].set_title('class')

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
