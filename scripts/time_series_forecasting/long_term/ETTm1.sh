model_name=TALON_GPT2

export CUDA_VISIBLE_DEVICES=7

# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_672_96 \
  --model $model_name \
  --data ETTm1 \
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
  --mlp_hidden_dim 256 \
  --mlp_hidden_layers 1 \
  --mlp_activation tanh \
  --topk 2 \
  --hidden_size 128 \
  --learning_rate 0.0002299591908114799 \
  --batch_size 256 \
  --lradj type1 \
  --weight_decay 1.6963819163410426e-06 \
  --alpha 0.1 \
  --beta 0.1 \


# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_672_96 \
  --model $model_name \
  --data ETTm1 \
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
  --mlp_hidden_dim 256 \
  --mlp_hidden_layers 1 \
  --mlp_activation tanh \
  --topk 2 \
  --hidden_size 128 \
  --learning_rate 0.0002299591908114799 \
  --batch_size 256 \
  --lradj type1 \
  --weight_decay 1.6963819163410426e-06 \
  --alpha 0.1 \
  --beta 0.1 \
  --test_dir long_term_forecast_ETTm1_672_96_TALON_GPT2_ETTm1_sl672_ll576_tl96_tpl96_lr0.0002299591908114799_alpha0.1_beta0.1_bt256_wd1.6963819163410426e-06_hd256_hl1_acttanh_k2_hs128_lradjtype1_cosTrue_test_0
done
