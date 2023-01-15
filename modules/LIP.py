import brica1
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cerenaut_pt_core.components.sparse_autoencoder import SparseAutoencoder

from modules.common.obs_convertion import (get_angles_and_image,
                                           get_flatten_observation,
                                           render_image_for_debug)


class LIP(brica1.PipeComponent):
    def __init__(self, observation_dim=1, token_dim=1, config=None):
        super(LIP, self).__init__()
        self.make_in_port('in', observation_dim)
        self.make_out_port('out', 1802)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)

        # オートエンコーダのセットアップ
        use_cuda = not config["no_cuda"] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.encoder = SparseAutoencoder((1, 1, 128, 128), SparseAutoencoder.get_default_config()).to(self.device)

    def fire(self):
        # フラット化されたベクトルが入力されるので、 Image部とAngle部に分ける
        in_data = self.get_in_port('in').buffer
        (image, angle_h, angle_v) = get_angles_and_image(in_data)

        # サリエンシーマップに変換
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        bool, map = saliency.computeSaliency(image.astype(np.float32))
        i_saliency = (map * 255).astype("uint8")
        render_image_for_debug(i_saliency, "LIP")

        # オートエンコーダで次元削減
        i_saliency = np.reshape(i_saliency,
                                     [1, 1, i_saliency.shape[0],
                                      i_saliency.shape[0]])
        data = torch.from_numpy(i_saliency.astype(np.float64)).float().to(self.device)
        encoding, _ = self.encoder(data)
        encodingNum = encoding.to('cpu').detach().numpy().copy()
        
        # 再度フラット化して出力する
        self.results['out'] = get_flatten_observation(encodingNum, angle_h, angle_v)
        self.results['token_out'] = self.inputs['token_in']