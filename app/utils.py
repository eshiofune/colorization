import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torchvision.models.resnet import resnet34
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.color import lab2rgb


class Discriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]
        model += [
            self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)
        ]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, init="norm", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and "Conv" in classname:
            if init == "norm":
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    print(f"Initializing the model with {init} initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


class GANLoss(nn.Module):
    def __init__(self, real_label=0.9, fake_label=0.1):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class Model(nn.Module):
    def __init__(
        self, G_net, lr_G=0.0004, lr_D=0.0004, beta1=0.5, beta2=0.999, lamda=100.0
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lamda = lamda

        self.G_net = G_net.to(self.device)
        self.D_net = init_model(
            Discriminator(input_c=3, n_down=3, num_filters=64), self.device
        )
        self.GANcriterion = GANLoss().to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.G_net.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.D_net.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.fake_color = self.G_net(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.D_net(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.D_net(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.D_net(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lamda
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.D_net.train()
        self.set_requires_grad(self.D_net, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.G_net.train()
        self.set_requires_grad(self.D_net, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


def build_generator(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = create_body(resnet34(), pretrained=True, n_in=n_input, cut=-2)
    G_net = DynamicUnet(backbone, n_output, (size, size)).to(device)
    return G_net


def load_models(checkpoint_paths: str):
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for checkpoint_id, checkpoint_path in checkpoint_paths.items():
        G_net = build_generator()
        model = Model(
            G_net=G_net, lr_G=0.0004, lr_D=0.0004, beta1=0.5, beta2=0.999, lamda=100.0
        )

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.G_net.load_state_dict(checkpoint["G_net_state_dict"])
        except FileNotFoundError as e:
            print(f"Error loading checkpoint: {e}")

        model.G_net.eval()
        models[checkpoint_id] = model

    return models


def preprocess_image(image: Image.Image):
    original_size = image.size
    image = image.resize((256, 256))
    image = (
        transforms.ToTensor()(image.convert("L")) * 2.0 - 1.0
    )  # Convert to L channel and normalize
    return (
        image.unsqueeze(0),
        original_size,
    )  # Add batch dimension and return original size


def postprocess_output(
    output: torch.Tensor, input_L: torch.Tensor, original_size: tuple
) -> Image.Image:
    output = output.squeeze(0).detach().cpu().numpy()
    L = input_L.squeeze(0).cpu().numpy() * 50.0 + 50.0  # Denormalize L channel
    ab = output * 110.0  # Denormalize ab channels
    lab_image = np.zeros((256, 256, 3))
    lab_image[..., 0] = L
    lab_image[..., 1:] = ab.transpose(1, 2, 0)
    rgb_image = lab2rgb(lab_image)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    colored_image = Image.fromarray(rgb_image)
    colored_image = colored_image.resize(original_size)  # Resize to original size
    return colored_image

