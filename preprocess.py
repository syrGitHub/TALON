import argparse
import torch
from models.Preprocess import Model

from data_provider.data_loader import Dataset_Patch_Preprocess
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')  # GPT2, Qwen-0.5B, Deepseek, LLAMA
    parser.add_argument('--llm_ckp_dir', type=str, default='./LLM/gpt2', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        help='dataset to preprocess, options:[ETTh1, electricity, weather, traffic]')
    args = parser.parse_args()
    print(args.dataset)

    model = Model(args)

    seq_len = 672
    label_len = 576
    pred_len = 96

    assert args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'weather', 'traffic']
    if args.dataset == 'ETTh1':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh1.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'ETTh2':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTh2.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'ETTm1':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTm1.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'ETTm2':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/ETT-small/',
            data_path='ETTm2.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'electricity':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/electricity/',
            data_path='electricity.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'weather':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/weather/',
            data_path='weather.csv',
            size=[seq_len, label_len, pred_len])
    elif args.dataset == 'traffic':
        data_set = Dataset_Patch_Preprocess(
            root_path='./dataset/traffic/',
            data_path='traffic.csv',
            size=[seq_len, label_len, pred_len])

    data_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=False,
    )


    print(len(data_set.data_stamp))
    print(data_set.tot_len)
    save_dir_path = './dataset/ETT-small/'
    output_list = []
    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc='Prompt Embedding'):
        # print(data)
        output = model(data)
        output_list.append(output.detach().cpu())

    result = torch.cat(output_list, dim=0)
    print(result.shape)

    save_path = os.path.join(save_dir_path, f'{args.dataset}_{args.llm_model}.pt')
    torch.save(result, save_path)
    print(f'Result saved to: {save_path}')