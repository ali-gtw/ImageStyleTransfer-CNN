import torch
from torchvision import transforms

class ImageTransform:

    def __init__(self, cfg):
        self.preparation = transforms.Compose([
            transforms.Scale(cfg.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
            transforms.Normalize(mean=cfg.DATA.IMAGENET_MEAN, std=[1, 1, 1]),
            transforms.Lambda(lambda x: x.mul_(255)),
        ])

        self.post_preparation1 = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1. / 255)),
            transforms.Normalize(mean=[(-1)*x for x in cfg.DATA.IMAGENET_MEAN], std=[1, 1, 1]),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
        ])

        self.post_preparation2 = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def post_preparation(self, tensor):
        transformed_tensor = self.post_preparation1(tensor)
        transformed_tensor[transformed_tensor > 1] = 1
        transformed_tensor[transformed_tensor < 0] = 0
        image = self.post_preparation2(transformed_tensor)
        return image
