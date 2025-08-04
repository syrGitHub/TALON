model_name=TALON_GPT2

export CUDA_VISIBLE_DEVICES=1
# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_672_96 \
  --model $model_name \
  --data ETTh2 \
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
  --mlp_activation tanh \
  --topk 2 \
  --hidden_size 256 \
  --learning_rate 0.0012628833185921126 \
  --batch_size 256 \
  --lradj type2 \
  --weight_decay 2.6789715245834197e-07 \
  --alpha 0.07 \
  --beta 0.08 \


# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --llm_model GPT2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_672_96 \
  --model $model_name \
  --data ETTh2 \
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
  --mlp_activation tanh \
  --topk 2 \
  --hidden_size 256 \
  --learning_rate 0.0012628833185921126 \
  --batch_size 256 \
  --lradj type2 \
  --weight_decay 2.6789715245834197e-07 \
  --alpha 0.07 \
  --beta 0.08 \
  --test_dir long_term_forecast_ETTh2_672_96_TALON_GPT2_ETTh2_sl672_ll576_tl96_tpl96_lr0.0012628833185921126_alpha0.07_beta0.08_bt256_wd2.6789715245834197e-07_hd1024_hl0_acttanh_k2_hs256_lradjtype2_cosTrue_test_0
done
