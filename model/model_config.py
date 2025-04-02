class ModelConfig:
    def __init__(self):
        self.nb_samp = 64000
        self.first_conv = 128
        self.in_channels = 1
        self.filts = [128, [128, 128], [128, 512], [512, 512]]
        self.blocks = [2, 4]
        self.nb_fc_node = 1024
        self.gru_node = 1024
        self.nb_gru_layer = 3
        self.nb_classes = 2
