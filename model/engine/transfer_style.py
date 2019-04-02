import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from model.meta_arch import GramMSELoss, GramMatrix
from util.logger import setup_logger

def do_transfer_style (
        cfg,
        model,
        image_transformer,
        content_image,
        style_image,
        device
):
    logger = setup_logger('style-transfer', False)
    logger.info("Start transferring.")

    # define layers, loss functions
    loss_layers = cfg.LOSS.STYLE_LAYERS + cfg.LOSS.CONTENT_LAYERS
    loss_functions = [GramMSELoss()] * len(cfg.LOSS.STYLE_LAYERS ) + \
                     [nn.MSELoss()] * len(cfg.LOSS.CONTENT_LAYERS)
    loss_functions = [loss_function.to(device) for loss_function in loss_functions]

    # weights settings
    weights = cfg.LOSS.STYLE_WEIGHTS + cfg.LOSS.CONTENT_WEIGHTS

    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in model(style_image, cfg.LOSS.STYLE_LAYERS)]
    content_targets = [A.detach() for A in model(content_image, cfg.LOSS.CONTENT_LAYERS)]
    targets = style_targets + content_targets

    # create optimizer, and optimize_image
    # optimize_image = Variable(torch.randn(content_image.size()).type_as(content_image.data),
    #                           requires_grad=True)
    optimize_image = Variable(content_image.data.clone(), requires_grad=True)
    optimizer = optim.LBFGS([optimize_image])

    max_iterations = cfg.LOSS.MAX_ITER
    log_show_iter = cfg.LOSS.LOG_ITER_SHOW
    iterations = [0]
    while iterations[0] < max_iterations:
        def closure():
            optimizer.zero_grad()
            outputs = model(optimize_image, loss_layers)
            layer_losses = [weights[a] * loss_functions[a](A, targets[a]) for a,A in enumerate(outputs)]
            loss = sum(layer_losses)
            loss.backward()
            iterations[0] += 1
            if iterations[0] % log_show_iter == (log_show_iter - 1):
                logger.info('Iteration: %d, loss: %f' % (iterations[0] + 1, loss.data))

            return loss

        optimizer.step(closure)

    out_image = image_transformer.post_preparation(optimize_image.data[0].cpu().squeeze())
    out_image.save(cfg.OUTPUT.DIR + cfg.OUTPUT.FILE_NAME)
