r"""CNNIQA Model.

Created by: https://github.com/lidq92/CNNIQA

Modified by: Chaofeng Chen (https://github.com/chaofengc)

Modification:
    - We use 3 channel RGB input.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class CNNIQA(nn.Module):
    r"""CNNIQA model.
    Args:
        ker_size (int): Kernel size.
        n_kers (int): Number of kernals.
        n1_nodes (int): Number of n1 nodes.
        n2_nodes (int): Number of n2 nodes.
        pretrained_model_path (String): Pretrained model path.

    Reference:
        Kang, Le, Peng Ye, Yi Li, and David Doermann. "Convolutional
        neural networks for no-reference image quality assessment."
        In Proceedings of the IEEE conference on computer vision and
        pattern recognition, pp. 1733-1740. 2014.

    """

    def __init__(
        self,
        ker_size=7,
        n_kers=50,
        n1_nodes=800,
        n2_nodes=800,
        pretrained_model_path=None,
    ):
        super(CNNIQA, self).__init__()

        self.conv1 = nn.Conv2d(3, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

        if pretrained_model_path is not None:
            self.load_pretrained_network(pretrained_model_path)

    def load_pretrained_network(self, model_path):
        print(f'Loading pretrained model from {model_path}')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        self.net.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        r"""Compute IQA using CNNIQA model.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of CNNIQA model.

        """
        h = self.conv1(x)

        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        return q
