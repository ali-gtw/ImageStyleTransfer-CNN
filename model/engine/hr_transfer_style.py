from torch.autograd import Variable

from data import ImageTransform
from model.engine.utils import transform_image, optimize
from util.logger import setup_logger


logger = setup_logger('style-transfer', False)


def do_hr_transfer_style(
        cfg,
        model,
        content_image,
        style_image,
        optimized_image,
        device,
):
    logger.info("Start transferring.")

    image_transformer = ImageTransform(cfg.HRDATA.IMG_SIZE, cfg.DATA.IMAGENET_MEAN)

    # transform images
    content_image = transform_image(image_transformer, content_image, device)
    style_image = transform_image(image_transformer, style_image, device)
    optimized_image = transform_image(optimized_image, style_image, device)
    optimized_image = Variable(optimized_image.type_as(content_image.data), requires_grad=True)

    optimized_image = optimize(model, content_image, style_image, optimized_image, cfg, cfg.HRLOSS.MAX_ITER)

    out_image = image_transformer.post_preparation(optimized_image.data[0].cpu().squeeze())
    out_image.save(cfg.OUTPUT.DIR + cfg.OUTPUT.HR_FILE_NAME)
    return out_image
