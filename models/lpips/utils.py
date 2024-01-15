from collections import OrderedDict

import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


# def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
def get_state_dict():
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v0.2/alex.pth'
        # + f'master/lpips/weights/v0.1/vgg.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location='cpu',
        # map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
