import torch.nn as nn


class LaneNet(nn.Module):

    # Initialization
    def __init__(self, args):
        super(LaneNet, self).__init__()

        # Store arguments
        self.args = args

        # Use gpu flag
        # TODO: use_cuda || cuda_available
        self.use_cuda = args['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

