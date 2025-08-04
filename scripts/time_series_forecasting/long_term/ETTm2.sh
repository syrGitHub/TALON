model_name=TALON_GPT2

export CUDA_VISIBLE_DEVICES=7

# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_672_96 \
  --model $model_name \
  --data ETTm2 \
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
  --mlp_hidden_layers 0 \
  --mlp_activation relu \
  --topk 3 \
  --hidden_size 256 \
  --learning_rate 0.00010000868343107064 \
  --batch_size 256 \
  --lradj type2 \
  --weight_decay 1.5306751868933282e-05 \
  --alpha 0.08 \
  --beta 0.04 \


# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_672_96 \
  --model $model_name \
  --data ETTm2 \
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
  --mlp_hidden_layers 0 \
  --mlp_activation relu \
  --topk 3 \
  --hidden_size 256 \
  --learning_rate 0.00010000868343107064 \
  --batch_size 256 \
  --lradj type2 \
  --weight_decay 1.5306751868933282e-05 \
  --alpha 0.08 \
  --beta 0.04 \
  --test_dir long_term_forecast_ETTm2_672_96_TALON_GPT2_ETTm2_sl672_ll576_tl96_tpl96_lr0.00010000868343107064_alpha0.08_beta0.04_bt256_wd1.5306751868933282e-05_hd1024_hl0_actrelu_k3_hs256_lradjtype2_cosTrue_test_0
done
done