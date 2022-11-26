import brica1


class SC(brica1.PipeComponent):
    def __init__(self, action_dim=1, token_dim=1):
        super(SC, self).__init__()
        self.make_in_port('in', action_dim)
        self.make_out_port('out', action_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)
        self.set_map("in", "out")
        self.set_map("token_in", "token_out")        