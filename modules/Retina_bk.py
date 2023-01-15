import json

import brica1
import numpy as np
import torch
from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder


class Retina(brica1.PipeComponent):
    def __init__(self, observation_dim=1, token_dim=1, modelp=False, config=None):
        super(Retina, self).__init__()
        self.make_in_port('in', observation_dim)
        self.make_out_port('out', observation_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)
        self.modelp = modelp
        input_shape = [-1, observation_dim]
        if not modelp:
            self.set_map("in", "out")
            self.set_map("token_in", "token_out")
        
        # modelを使用する場合の設定
        retina_model = None
        if modelp:
            use_cuda = not config["no_cuda"] and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            if config['model_name'] == 'SimpleAE':
                retina_model = SimpleAutoencoder(input_shape, config['model_config']).to(device)
            else:
                raise NotImplementedError('Model not supported: ' + str(config['model_name']))
            retina_model.load_state_dict(torch.load(config["model_file"]))
            retina_model.eval()
            self.device = device
        self.model = retina_model
    
    def fire(self):
        if self.modelp:
            in_data = self.get_in_port('in').buffer
            if in_data.dtype == (np.dtype('int16') or np.dtype('float64')):
                in_data = np.zeros(self.in_dim, dtype='float64')
            x = in_data.reshape(1, len(in_data))
            data = torch.from_numpy(x.astype(np.float64)).float().to(self.device)
            encoding, _ = self.model(data)
            self.results['out'] = encoding.to(self.device).detach().numpy().reshape(len(in_data),)
            self.results['token_out'] = self.inputs['token_in']
        else:
            super(Retina, self).fire()
