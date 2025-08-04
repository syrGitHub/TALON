from models import TALON_GPT2


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TALON_GPT2': TALON_GPT2,
            'TALON_Qwen-0.5B': TALON_GPT2,
            'TALON_Deep-1.5B': TALON_GPT2,
            'TALON_LLAMA': TALON_GPT2,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
