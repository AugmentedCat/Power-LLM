
class GenerateConfig(object):
    def __init__(self,
                 seq_len: int,
                 nucleus_prob: float,
                 use_gpu: bool,
                 temperature: float = 1.0):
        self.seq_len = seq_len
        self.nucleus_prob = nucleus_prob
        self.use_gpu = use_gpu
        self.temperature = temperature
