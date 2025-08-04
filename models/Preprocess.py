import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        self.llm_model = configs.llm_model
        print(self.device)

        if self.llm_model == 'LLAMA':
            print(self.llm_model)
            self.llm = LlamaForCausalLM.from_pretrained(
                '/export/home2/n2409817j/LLM/Llama-2-7b',
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            self.tokenizer = LlamaTokenizer.from_pretrained('/export/home2/n2409817j/LLM/Llama-2-7b')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = self.tokenizer.vocab_size
            self.hidden_dim_of_llm = 4096

        if self.llm_model == 'GPT2':
            print(self.llm_model)
            self.llm = GPT2Model.from_pretrained(
                '/export/home2/n2409817j/LLM/gpt2',
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained('/export/home2/n2409817j/LLM/gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = self.tokenizer.vocab_size
            self.hidden_dim_of_llm = 768

        if self.llm_model == 'Qwen-0.5B':
            print(self.llm_model)
            self.llm = AutoModelForCausalLM.from_pretrained(
                '/export/home2/n2409817j/LLM/Qwen2-0.5B-Instruct',
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained('/export/home2/n2409817j/LLM/Qwen2-0.5B-Instruct')

        if self.llm_model == 'Deepseek':
            print(self.llm_model)
            self.llm = AutoModelForCausalLM.from_pretrained(
                '/export/home2/n2409817j/LLM/Deepseek-1.5B',
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained('/export/home2/n2409817j/LLM/Deepseek-1.5B')

        for name, param in self.llm.named_parameters():
            param.requires_grad = False

    def tokenize_input(self, x):
        output = self.tokenizer(x, return_tensors="pt", max_length=220, truncation=True,
                                padding=True)['input_ids'].to(self.device)
        # print(output.shape)  # torch.Size([1, 220])
        # 绝对长度保证
        if output.shape[1] != 220:
            padding = torch.full(
                (output.shape[0], 220 - output.shape[1]),
                self.tokenizer.pad_token_id,
                device=self.device
            )
            output = torch.cat([output, padding], dim=1)
        # 检查长度
        if output.shape[1] != 220:
            print("发现异常batch，样本示例:", x[0])  # 打印第一个样本
            print("实际长度:", output.shape[1])
        result = self.llm.get_input_embeddings()(output)
        return result
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc = torch.cat([self.tokenize_input(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        if self.llm_model == 'LLAMA':
            text_outputs = self.llm.model(inputs_embeds=x_mark_enc)[0]

        if self.llm_model == 'GPT2':
            text_outputs = self.llm(inputs_embeds=x_mark_enc)[0]

        if self.llm_model in ['Qwen-0.5B', 'Deepseek']:
            text_outputs = self.llm(inputs_embeds=x_mark_enc, output_hidden_states=True, return_dict=True).hidden_states[-1]
        print(text_outputs.shape)   # torch.Size([224, 220, 896])

        text_outputs = text_outputs[:, -1, :]
        print(text_outputs.shape)   # torch.Size([224, 896])
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)