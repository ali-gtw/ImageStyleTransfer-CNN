from torch import optim
from torch.autograd import Variable

from model.meta_arch import GramMatrix
from util.logger import setup_logger

logger = setup_logger('style-transfer', False)

def transform_image(image_transformer, image, device):
    image_transformed = image_transformer.preparation(image)
    image_transformed = Variable(image_transformed.unsqueeze(0).to(device))
    return image_transformed


def optimize(model, content_image, style_image, optimized_image, cfg, max_iterations):
    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in model.vgg_model(style_image, cfg.LOSS.STYLE_LAYERS)]
    content_targets = [A.detach() for A in model.vgg_model(content_image, cfg.LOSS.CONTENT_LAYERS)]
    targets = style_targets + content_targets

    # create optimizer
    optimizer = optim.LBFGS([optimized_image])

    log_show_iter = cfg.LOSS.LOG_ITER_SHOW * cfg.LOSS.MAX_ITER
    iterations = [0]
    while iterations[0] < max_iterations:
        def closure():
            optimizer.zero_grad()
            outputs = model.vgg_model(optimized_image, model.loss_layers)
            layer_losses = [model.loss_weights[a] * model.loss_functions[a](A, targets[a]) for a, A in
                            enumerate(outputs)]
            loss = sum(layer_losses)
            loss.backward()
            iterations[0] += 1
            if iterations[0] % log_show_iter == (log_show_iter - 1):
                logger.info('Iteration: %d, loss: %f' % (iterations[0] + 1, loss.data))

            return loss

        optimizer.step(closure)
    return optimized_image
