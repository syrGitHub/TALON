import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.expert_extractor_cluster import Heterogeneous_MoE
from transformers import LlamaForCausalLM, LlamaTokenizer, GPT2Model, GPT2Tokenizer

from layers.mlp import MLP
import time


class ContrastiveHead(nn.Module):
    def __init__(self, hidden_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),  # 添加BN
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x1, x2):
        x1 = F.normalize(self.proj(x1), dim=-1)
        x2 = F.normalize(self.proj(x2), dim=-1)
        return torch.matmul(x1, x2.T) / self.temperature  # [B*N, B*N]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # load parameters
        self.seq_len = configs.test_seq_len
        self.pred_len = configs.test_pred_len
        self.token_len = configs.token_len
        self.expert_num = configs.num_experts
        self.llm_model = configs.llm_model
        self.training = configs.is_training
        self.alpha = configs.alpha

        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)

        self.add_interpatch_scale = nn.Parameter(torch.ones([]))

        # Load LLaMA model and tokenizer
        if self.llm_model == 'LLAMA':
            start = time.time()
            self.llm = LlamaForCausalLM.from_pretrained(
                '/export/home2/n2409817j/LLM/Llama-2-7b',
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            print("Loaded LLM model in", time.time() - start)
            self.tokenizer = LlamaTokenizer.from_pretrained('/export/home2/n2409817j/LLM/Llama-2-7b')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = self.tokenizer.vocab_size
            self.hidden_dim_of_llm = 4096

        if self.llm_model == 'GPT2':
            start = time.time()
            self.llm = GPT2Model.from_pretrained(
                '/export/home2/n2409817j/LLM/gpt2',
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            print("Loaded LLM model in", time.time() - start)
            self.tokenizer = GPT2Tokenizer.from_pretrained('/export/home2/n2409817j/LLM/gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = self.tokenizer.vocab_size
            self.hidden_dim_of_llm = 768

        for name, param in self.llm.named_parameters():
            param.requires_grad = False

        self.cluster = Heterogeneous_MoE(configs, self.token_len, self.hidden_dim_of_llm, configs.num_experts,
                                         configs.hidden_size, configs.hidden_size, True, configs.topk)

        self.contrastive_head = ContrastiveHead(self.hidden_dim_of_llm)
        self.contrastive_loss = nn.CrossEntropyLoss()

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as detokenizer")
            self.decoder = nn.Linear(self.hidden_dim_of_llm, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as detokenizer")
            self.decoder = MLP(self.hidden_dim_of_llm, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, complexity):
        # print('x_mark_enc.shape', x_mark_enc.shape) # torch.Size([256, 7, 4096])
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape  # torch.Size([256, 672, 1])
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        B_N = fold_out.shape[0]
        token_num = fold_out.shape[1]
        times_embeds = fold_out.reshape(-1, self.token_len)
        complexity = complexity.reshape(-1, self.expert_num)

        # times_embeds: [bs * n_vars * token_num x token_len] complexity: [bs * n_vars * token_num x expert_num]
        times_embeds, L_importance, gates, load = self.cluster(times_embeds, complexity, self.alpha)
        times_embeds = times_embeds.view(B_N, token_num, self.hidden_dim_of_llm)

        # 对比学习部分
        loss_contrast = 0.0
        if self.training and x_mark_enc is not None:  # 仅在训练时使用对比学习
            # 对齐 times_embeds 和 x_mark_enc 的维度
            times_embeds_flat = times_embeds.view(-1, self.hidden_dim_of_llm)  # [B*N*T, D]
            x_mark_enc_flat = x_mark_enc.view(-1, self.hidden_dim_of_llm)  # [B*N*T, D]

            # 计算对比损失
            logits = self.contrastive_head(times_embeds_flat, x_mark_enc_flat)
            labels = torch.arange(logits.size(0)).to(self.device)  # 对角线是正样本
            loss_contrast = self.contrastive_loss(logits, labels)

        # inter-patch modelling
        # outputs: [bs * n_vars x token_num x hidden_dim_of_llm]
        if self.llm_model == 'LLAMA':
            outputs = self.llm.model(inputs_embeds=times_embeds).last_hidden_state
        if self.llm_model == 'GPT2':
            outputs = self.llm(inputs_embeds=times_embeds).last_hidden_state

        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out, L_importance, loss_contrast # , gates, load, complexity

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, complexity):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, complexity)

