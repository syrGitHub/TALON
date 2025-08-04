model_name=TALON_GPT2

export CUDA_VISIBLE_DEVICES=4

# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --train_epochs 10 \
  --gpu 0 \
  --tmax 10 \
  --drop_last \
  --use_amp \
  --cosine \
  --visualize \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_layers 1 \
  --mlp_activation tanh \
  --topk 3 \
  --hidden_size 1024 \
  --learning_rate 0.0006590475277295834 \
  --batch_size 384 \
  --lradj type1 \
  --weight_decay 6.209452255793491e-07 \
  --alpha 0.02 \
  --beta 0.1

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --train_epochs 10 \
  --gpu 0 \
  --tmax 10 \
  --drop_last \
  --use_amp \
  --cosine \
  --visualize \
  --mlp_hidden_dim 1024 \
  --mlp_hidden_layers 1 \
  --mlp_activation tanh \
  --topk 3 \
  --hidden_size 1024 \
  --learning_rate 0.0006590475277295834 \
  --batch_size 384 \
  --lradj type1 \
  --weight_decay 6.209452255793491e-07 \
  --alpha 0.02 \
  --beta 0.1 \
  --test_dir long_term_forecast_ETTh1_672_96_TALON_GPT2_ETTh1_sl672_ll576_tl96_tpl96_lr0.0006590475277295834_alpha0.02_beta0.1_bt384_wd6.209452255793491e-07_hd1024_hl1_acttanh_k3_hs1024_lradjtype1_cosTrue_test_0
done