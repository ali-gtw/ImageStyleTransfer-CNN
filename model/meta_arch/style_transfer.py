
class StyleTransfer:
    def __init__(self,  vgg_model, loss_layers, loss_functions, loss_weights):
        self.vgg_model = vgg_model
        self.loss_layers = loss_layers
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
