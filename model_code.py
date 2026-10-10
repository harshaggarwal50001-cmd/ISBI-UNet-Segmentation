import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum
from torch.utils.data import Dataset, DataLoader

from einops import rearrange
from einops.layers.torch import Rearrange

from PIL import Image
import torchvision.transforms.functional as TF


def if_exist(val):
    return val is not None

def default(val, def_val):
    if if_exist(val):
        return val
    return def_val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsampling(dim_in, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim_in, dim_out, 3, padding=1)
    )

def Downsampling(dim_in, dim_out):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim_in * 4, dim_out, 3, padding=1)
    )

class Weight_standard(nn.Conv2d):
    def forward(self, x):
        weights = self.weight
        bias = self.bias
        mean = weights.mean(dim=[1, 2, 3], keepdim=True)
        var = weights.var(dim=[1, 2, 3], keepdim=True)
        weights = (weights - mean) / (torch.sqrt(var + 1e-5))
        return F.conv2d(x, weights, bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.normalize_weight = Weight_standard(dim_in, dim_out, 3, padding=1)
        self.group_norm = nn.GroupNorm(num_channels=dim_out, num_groups=groups)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.normalize_weight(x)
        h = self.group_norm(h)
        y = self.act(h)
        return y


class ResNet(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.b1 = Block(dim_in, dim_out, groups=groups)
        self.b2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x):
        h = self.b1(x)
        h = self.b2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_head=32, head=4):
        super().__init__()
        self.scale = 1 / (dim_head ** 0.5)
        self.head = head
        self.dim_head = dim_head
        self.hidden_dim = head * dim_head
        self.to_qvk = nn.Conv2d(dim_in, self.hidden_dim * 3, 3, padding=1)
        self.rearrange_qvk = Rearrange("b (h c) x y -> b c (x y) h", h=self.dim_head)
        self.to_out_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, dim_out, 3, padding=1),
            nn.GroupNorm(1, dim_out)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.to_qvk(x).chunk(3, dim=1)
        q, v, k = map(self.rearrange_qvk, x)
        q = q * self.scale
        attn = einsum("b c i j, b c d j -> b c i d", q, k)
        attn = attn - attn.amax(dim=-1, keepdim=True).detach()
        attention = attn.softmax(dim=-1)
        out = einsum("b c i j, b c j d -> b c i d", attention, v)
        out = rearrange(out, "b c (x y) a -> b (c a) x y", x=h, y=w)
        return self.to_out_conv(out)

class LinearAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_head=32, head=4):
        super().__init__()
        self.scale = 1 / (dim_head ** 0.5)
        self.head = head
        self.dim_head = dim_head
        self.hidden_dim = head * dim_head
        self.to_qvk = nn.Conv2d(dim_in, self.hidden_dim * 3, 3, padding=1)
        self.rearrange_qvk = Rearrange("b (h c) x y -> b c (x y) h", h=self.dim_head)
        self.to_out_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, dim_out, 3, padding=1),
            nn.GroupNorm(1, dim_out)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qvk = self.to_qvk(x)
        q, v, k = map(self.rearrange_qvk, qvk.chunk(3, dim=1))
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * self.scale
        context = einsum("b c i j, b c i d -> b c j d", k, v)
        attn = einsum("b c d e, b c e n -> b c d n", q, context)
        out = rearrange(attn, "b c (x y) d -> b (c d) x y", x=h, y=w)
        return self.to_out_conv(out)


class PreNorm(nn.Module):
    def __init__(self, fn, dim_in, groups=8):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim_in)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class UNet(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8, mults=(1, 2, 4, 8), init_dim=None, out_dim=None):
        super().__init__()
        self.init_dim = default(init_dim, dim_in)
        self.groups = groups
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.init_conv = nn.Conv2d(dim_in, self.init_dim, 3, padding=1)

        dims = [self.init_dim, *map(lambda t: t * self.init_dim, mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for index, (d_in, d_out) in enumerate(in_out):
            is_last = index == (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList([
                    ResNet(d_in, d_in, groups=groups),
                    ResNet(d_in, d_in, groups=groups),
                    Residual(PreNorm(LinearAttention(d_in, d_in), d_in, groups=groups)),
                    Downsampling(d_in, d_out) if not is_last else nn.Conv2d(d_in, d_out, 3, padding=1)
                ])
            )

        mid_dim = dims[-1]
        self.mid_layer1 = ResNet(mid_dim, mid_dim, groups=groups)
        self.mid_attention = Residual(PreNorm(SelfAttention(mid_dim, mid_dim), mid_dim, groups=groups))
        self.mid_layer2 = ResNet(mid_dim, mid_dim, groups=groups)

        for index, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = index == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    ResNet(d_in + d_out, d_out, groups=groups),
                    ResNet(d_in + d_out, d_out, groups=groups),
                    Residual(PreNorm(LinearAttention(d_out, d_out), d_out, groups=groups)),
                    Upsampling(d_out, d_in) if not is_last else nn.Conv2d(d_out, d_in, 3, padding=1)
                ])
            )

        self.final_resnet_block = nn.Conv2d(d_in * 2, d_in, 3, padding=1)
        self.final_convlayer = nn.Conv2d(d_in, self.dim_out, 3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        r = x.clone()
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)
            x = block2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_layer1(x)
        x = self.mid_attention(x)
        x = self.mid_layer2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_resnet_block(x)
        return self.final_convlayer(x)


if __name__ == "__main__":
    import zipfile
    from sklearn.model_selection import train_test_split

    BASE_DIR = "/content/isbi2012"

    TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "unmodified-data", "train", "imgs")
    TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "unmodified-data", "train", "labels")

    TEST_IMAGE_DIR = os.path.join(BASE_DIR, "unmodified-data", "test", "imgs")
    TEST_LABEL_DIR = os.path.join(BASE_DIR, "unmodified-data", "test", "labels")

    print("Train images:", len(os.listdir(TRAIN_IMAGE_DIR)))
    print("Train labels:", len(os.listdir(TRAIN_LABEL_DIR)))
    print("Test images:", len(os.listdir(TEST_IMAGE_DIR)))
    print("Test labels:", len(os.listdir(TEST_LABEL_DIR)))

    PATCH_SIZE = 128
    STRIDE = 64

    def extract_patches(image, label, patch_size, stride):
        patch_images = []
        patch_labels = []
        image_array = np.array(image)
        label_array = np.array(label)
        h, w = image_array.shape[:2]
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch_image = image_array[i:i+patch_size, j:j+patch_size]
                patch_label = label_array[i:i+patch_size, j:j+patch_size]
                patch_images.append(patch_image)
                patch_labels.append(patch_label)
        return patch_images, patch_labels

    def create_patches(image_names, label_names, image_dir, label_dir):
        image_patches = []
        label_patches = []
        for img_name, label_name in zip(image_names, label_names):
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, label_name)
            image = Image.open(img_path)
            label = Image.open(label_path)
            patches_img, patches_label = extract_patches(image, label, patch_size=PATCH_SIZE, stride=STRIDE)
            image_patches.extend(patches_img)
            label_patches.extend(patches_label)
        return image_patches, label_patches

    all_train_images = sorted(os.listdir(TRAIN_IMAGE_DIR))
    all_train_labels = sorted(os.listdir(TRAIN_LABEL_DIR))

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        all_train_images, all_train_labels, test_size=0.2, random_state=42
    )

    train_image_patches, train_label_patches = create_patches(train_imgs, train_lbls, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    val_image_patches, val_label_patches = create_patches(val_imgs, val_lbls, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)

    class BasicTransform:
        def __call__(self, image, label):
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            return image, label

    class ISBIDataset(Dataset):
        def __init__(self, image_patches, label_patches, transform=None):
            self.image_patches = image_patches
            self.label_patches = label_patches
            self.transform = transform
        def __len__(self):
            return len(self.image_patches)
        def __getitem__(self, idx):
            image = self.image_patches[idx]
            label = self.label_patches[idx]
            image = Image.fromarray(image)
            label = Image.fromarray(label)
            if self.transform is not None:
                image, label = self.transform(image, label)
            return image, label

    basic_transform = BasicTransform()
    train_dataset = ISBIDataset(train_image_patches, train_label_patches, transform=basic_transform)
    val_dataset = ISBIDataset(val_image_patches, val_label_patches, transform=basic_transform)

    batch_size = 4
    num_workers = 2
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(dim_in=1, dim_out=1, init_dim=32, mults=(1, 2, 4, 8))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    num_epochs = 30
    max_grad_norm = 1.0
    best_val_loss = float('inf')

    print("Start..")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_dataloader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"Saved Model {best_val_loss:.4f}")

    print("DONE")