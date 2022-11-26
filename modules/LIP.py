import brica1


class LIP(brica1.PipeComponent):
    def __init__(self, observation_dim=1, token_dim=1):
        super(LIP, self).__init__()
        self.make_in_port('in', observation_dim)
        self.make_out_port('out', observation_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)
        self.set_map("in", "out")
        self.set_map("token_in", "token_out")        