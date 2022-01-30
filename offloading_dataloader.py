# This is a simplified version of the Active MNIST data loader

import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import idx2numpy
import torch
import torch.nn.functional as func
import torchvision.transforms


def flatten_channel_dim(imgs):
    imgs = imgs.cpu()
    return torch.reshape(imgs, [imgs.shape[0], imgs.shape[2], imgs.shape[3]])


def to_one_hot(y, dims=10, dtype=torch.FloatTensor):
    out = torch.zeros((len(y), dims)).type(dtype)
    y = y.type(torch.LongTensor)

    for i in range(len(out)):
        out[i, y[i]] = 1

    return out


def plot_example(original_images, transformed_images, labels, transforms, number_examples=1):
    assert len(original_images) == len(transformed_images) == len(labels) == len(transforms)
    assert len(original_images) >= number_examples

    for i in range(number_examples):
        fig, axs = plt.subplots(ncols=2, figsize=(6, 3))
        axs[0].imshow(flatten_channel_dim(original_images)[i], cmap=cm.gray_r)
        axs[0].set_title('Original MNIST Figure')
        axs[1].imshow(flatten_channel_dim(transformed_images)[i], cmap=cm.gray_r)
        axs[1].set_title('Transformed MNIST Figure')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        label = torch.where(labels[i] == 1)[0]

        description = f'Label = {label.data.cpu().numpy()}\n' \
                      f'Scale = {transforms[i, 0].data.cpu().numpy():.3f}\n' \
                      f'Rotation = {transforms[i, 1].data.cpu().numpy() / math.pi:.3f}π'
        plt.text(0.95, 0.5, description, fontsize=14, transform=plt.gcf().transFigure)
        plt.show()


class Dataloader:

    def __init__(self, image_path, label_path, batch_size=1024, max_iterations=64, size=56,
                 n_of_each=None, shuffle=True, dtype=None):

        if dtype is None:
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        else:
            self.dtype = dtype

        self.shuffle = shuffle

        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        if n_of_each is not None:
            images_filtered = np.empty(shape=(n_of_each * 10, 28, 28), dtype=None)
            labels_filtered = np.zeros(shape=n_of_each * 10)

            for i in range(10):
                imgs_tmp = images[labels == i]
                indices = np.random.randint(0, len(imgs_tmp), size=n_of_each)
                images_filtered[i * n_of_each:(i + 1) * n_of_each] = imgs_tmp[indices]
                labels_filtered[i * n_of_each:(i + 1) * n_of_each] = np.ones(shape=n_of_each) * i

            images = images_filtered
            labels = labels_filtered

        # add channel dimension
        self.images = torch.Tensor(np.reshape(images, [images.shape[0], 1, images.shape[1], images.shape[2]]))
        self.labels = torch.Tensor(labels)

        # shuffle
        indices = torch.randperm(len(self.images))
        self.images = self.images[indices]
        self.labels = self.labels[indices]

        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.iter = 0

        self.n_of_each = n_of_each
        self.size = size

        self.mean_scale = 1
        self.std_scale = 0.25
        self.mean_rotation = 0
        self.std_rotation = math.pi / 2

        self.or_img = None
        self.trns_img = None
        self.lbl = None
        self.trns = None

        print(f'Active MNIST Dataloader initialized on {self.dtype}\n'
              f'batch_size = {self.batch_size}, iterations = {self.max_iterations}')

    def load_data(self, perc_normal=0.0, perc_distractors=0.0):
        del self.or_img, self.trns_img, self.lbl, self.trns
        if self.dtype == torch.cuda.FloatTensor:
            torch.cuda.empty_cache()

        print(f'Sampling {self.batch_size * self.max_iterations} images...')
        print(f' {(1 - (perc_normal + perc_distractors)) * 100}% transformed\n {(perc_distractors) * 100}% distractors\n {(perc_normal) * 100}% normal.')
        self.or_img, self.trns_img, self.lbl, self.trns = self.sample(n=self.batch_size * self.max_iterations,
                                                                      perc_normal=perc_normal,
                                                                      perc_distractors=perc_distractors)
        if self.shuffle:
            self.shuffle_data()
        print(f'Finished sampling. Loaded {self.batch_size * self.max_iterations} images into {self.dtype}.')

    def set_scale(self, mean=1, std=1):
        self.mean_scale = mean
        self.std_scale = std

    def set_rotation(self, mean=0, std=math.pi / 2):
        self.mean_rotation = mean
        self.std_rotation = std

    def shuffle_data(self):
        assert self.or_img is not None
        assert self.trns_img is not None
        assert self.lbl is not None
        assert self.trns is not None

        indices = torch.randperm(len(self.or_img))
        self.or_img = self.or_img[indices]
        self.trns_img = self.trns_img[indices]
        self.lbl = self.lbl[indices]
        self.trns = self.trns[indices]

    def __iter__(self):
        return self

    def __next__(self):
        assert self.or_img is not None, 'Call load_data() before iterating!'

        if self.iter < self.max_iterations:
            out = self.or_img[self.iter * self.batch_size: (self.iter + 1) * self.batch_size], \
                  self.trns_img[self.iter * self.batch_size: (self.iter + 1) * self.batch_size], \
                  self.lbl[self.iter * self.batch_size: (self.iter + 1) * self.batch_size], \
                  self.trns[self.iter * self.batch_size: (self.iter + 1) * self.batch_size]
            self.iter += 1
            return out
        else:
            self.iter = 0
            if self.shuffle:
                self.shuffle_data()
            raise StopIteration

    def sample(self, n=1, perc_normal=0.0, perc_distractors=0.0):
        assert 0.0 <= perc_normal + perc_distractors <= 1, \
            'Percentage of normal/untransformed + distractor images must be between 0 and 1!'

        def transformation_matrix(transform=[1, 0]):
            # scale
            T_scale = torch.Tensor([[transform[0], 0, 0],
                                    [0, transform[0], 0],
                                    [0, 0, 1]]).type(self.dtype)
            # rotation
            T_rot = torch.Tensor([[torch.cos(transform[1]), -torch.sin(transform[1]), 0],
                                  [torch.sin(transform[1]), torch.cos(transform[1]), 0],
                                  [0, 0, 1]]).type(self.dtype)

            # combine matrices
            T = T_rot @ T_scale
            T_inv = torch.linalg.inv(T).type(self.dtype)
            return T_inv[0:2, :]


        n_normal = int(n * perc_normal)
        n_distractors = int(n * perc_distractors)
        n_transformed = int(n - (n_normal + perc_distractors))

        print('     Sampling indices')
        index_trns = np.random.choice(len(self.images), size=n_transformed, replace=True)
        index_normal = np.random.choice(len(self.images), size=n_normal, replace=True)
        index_dist = np.random.choice(len(self.images), size=n_distractors, replace=True)

        print('     Creating random transformation values')
        if n_transformed > 0:
            scales = torch.Tensor(np.abs(np.random.normal(loc=self.mean_scale,
                                                          scale=self.std_scale,
                                                          size=(n_transformed, 1)))).type(self.dtype)
            rotations = torch.Tensor((np.abs(np.random.normal(loc=self.mean_rotation,
                                                              scale=self.std_rotation,
                                                              size=(n_transformed, 1)))
                                      % math.pi)
                                     * np.random.choice([-1, 1])).type(self.dtype)

            trns = torch.cat((scales, rotations), dim=1).type(self.dtype)
        else:
            trns = torch.zeros(n_transformed, 2).type(self.dtype)

        if n_distractors > 0:
            scales = torch.Tensor(np.abs(np.random.normal(loc=1, scale=0.25,
                                                          size=(n_distractors, 1)))).type(self.dtype)
            rotations = torch.Tensor((np.abs(np.random.normal(loc=0, scale=math.pi / 2,
                                                              size=(n_distractors, 1)))
                                      % math.pi)
                                     * np.random.choice([-1, 1])).type(self.dtype)

            trns_dist = torch.cat((scales, rotations), dim=1).type(self.dtype)
        else:
            trns_dist = torch.zeros(n_distractors, 2).type(self.dtype)

        trns_normal = torch.zeros(n_normal, 2).type(self.dtype)

        # select images and labels based on indices, only copy these to *device* (not whole dataset)
        selected_imgs_trns = self.images[index_trns].type(self.dtype)
        lbl_trns = to_one_hot(torch.Tensor(self.labels[index_trns]).type(self.dtype), dtype=self.dtype)

        selected_imgs_dist = self.images[index_dist].type(self.dtype)  # TODO: (maybe) add perlin noise
        lbl_dist = func.softmax(
            torch.add(
                torch.add(
                    torch.randn(size=[n_distractors, 10]).type(self.dtype),
                    to_one_hot(torch.Tensor(self.labels[index_dist]).type(self.dtype), dtype=self.dtype),
                    alpha=2),
                10)/10
            , dim=1)  # makes class distribution mostly flat with a bit of noise and a small peak at true label

        selected_imgs_normal = self.images[index_normal].type(self.dtype)
        lbl_normal = to_one_hot(torch.Tensor(self.labels[index_normal]).type(self.dtype), dtype=self.dtype)

        # padding images
        print('     Padding images')
        or_img_trns = torchvision.transforms.Pad(padding=(self.size - self.images.shape[2]) // 2)(selected_imgs_trns)
        or_img_normal = torchvision.transforms.Pad(padding=(self.size - self.images.shape[2]) // 2)(
            selected_imgs_normal)
        or_img_dist = torchvision.transforms.Pad(padding=(self.size - self.images.shape[2]) // 2)(
            selected_imgs_dist)

        # transforming images
        if n_transformed > 0:
            print('     Transforming images')
            trns_matrices = torch.stack([transformation_matrix(t) for t in trns]).type(self.dtype)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid = func.affine_grid(trns_matrices, or_img_trns.size()).type(self.dtype)
                trns_img_trns = func.grid_sample(or_img_trns, grid).type(self.dtype)
        else:
            trns_img_trns = torch.zeros((n_transformed, 1, self.size, self.size)).type(self.dtype)

        if n_distractors > 0:
            print('     Transforming distractors')
            trns_matrices = torch.stack([transformation_matrix(t) for t in trns_dist]).type(self.dtype)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid = func.affine_grid(trns_matrices, or_img_dist.size()).type(self.dtype)
                trns_img_dist = func.grid_sample(or_img_dist, grid).type(self.dtype)
        else:
            trns_img_dist = torch.zeros((n_distractors, 1, self.size, self.size)).type(self.dtype)

        trns_img_normal = torch.clone(or_img_normal).type(self.dtype)

        # concatenating transformed, distractor and normal data
        or_img_trns = torch.concat((or_img_normal, or_img_trns, or_img_dist), dim=0).type(self.dtype)
        trns_img_trns = torch.concat((trns_img_normal, trns_img_trns, trns_img_dist), dim=0).type(self.dtype)
        lbl_trns = torch.concat((lbl_normal, lbl_trns, lbl_dist), dim=0).type(self.dtype)
        trns = torch.concat((trns_normal, trns, trns_dist), dim=0).type(self.dtype)
        # Using "trns" variables to save gpu memory. Those usually make up the majority.

        return or_img_trns, trns_img_trns, lbl_trns, trns
