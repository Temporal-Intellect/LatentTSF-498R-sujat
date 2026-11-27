#!/bin/bash
# Meta-correction in the latent residual space (498R-kdd style) of one trained LatentTSF forecaster:
#   Y_hat = D( Ẑ_Y − alpha · f_phi(R[t − L_m : t]) ),   R = Ẑ_Y − Z_Y  (from run_residual.sh)
# f_phi is trained on the residual series with the base model's splits, alpha is chosen on validation.
#
# Run order, all with the same command-line flags (`bash run_meta.sh --help`):
#   bash run_train.sh  ->  bash run_residual.sh  ->  bash run_meta.sh
# The python flags below must stay identical to run_train.sh: they locate the base checkpoint and residuals.
# By default f_phi mirrors the base forecaster (same backbone, seq_len and hyperparameters); change it with
# --meta_args, e.g. bash run_meta.sh --meta_args model=iTransformer seq_len=96 d_model=128
# Outputs go to ./residuals/<model_id>/meta_<loss>[_<overrides>]/ (see README, "Meta-correction").

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============ User-tunable: defaults, override them on the command line ============
dataset="ETTh1"
model="DLinear"
pred_len=96
meta_args=()      # overrides of f_phi's mirrored config, e.g. (model=iTransformer seq_len=96)
meta_loss="huber" # huber (498R-kdd) or mse
# ====================================================================================

usage() {
  cat <<'EOF'
Usage: bash run_meta.sh [--dataset NAME] [--model NAME] [--pred_len N]
                        [--meta_args KEY=VALUE ...] [--meta_loss huber|mse] [flags ...]

  --dataset NAME             ETTh1 (default), ETTh2, ETTm1, ETTm2, weather, exchange_rate, electricity, traffic
  --model NAME               backbone of the forecaster trained by run_train.sh (default: DLinear)
  --pred_len N               its forecast horizon (default: 96)
  --meta_args KEY=VALUE ...  overrides of f_phi's config, which otherwise mirrors the forecaster, using
                             my_train.py flag names (e.g. model=iTransformer seq_len=96 d_model=128)
  --meta_loss huber|mse      training loss of f_phi (default: huber)

Repeat every other flag you gave run_train.sh: the checkpoint and residuals are found from them.
Other flags go to my_meta.py (python my_meta.py --help), which also takes:
  --alpha_grid A,B,...    candidate alphas (default 0,0.05,...,1; 0 and 0.05 are always added)
  --huber_delta D         Huber threshold (default 1.0)
  --meta_train 0          evaluate the saved f_phi instead of training it again
  --save_final_npy 0      skip the decoded test .npy files
  --residual_dir DIR, --residual_name NAME   as given to run_residual.sh

Environment: CUDA_VISIBLE_DEVICES (default 0).

Examples:
  bash run_meta.sh
  bash run_meta.sh --dataset ETTh2 --model iTransformer --pred_len 192 --meta_loss mse
  bash run_meta.sh --dataset weather --pred_len 336 --learning_rate 0.0005 --train_epochs 50 \
                   --meta_args model=iTransformer seq_len=96 d_model=128
  bash run_meta.sh --meta_train 0 --alpha_grid 0,0.02,0.05,0.1
EOF
}

# --dataset / --model / --pred_len / --meta_args / --meta_loss are read here; every other argument goes to my_meta.py
extra_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dataset=*|--model=*|--pred_len=*|--meta_loss=*|--meta_args=*) set -- "${1%%=*}" "${1#*=}" "${@:2}"; continue ;;
    --meta_args)  # every following KEY=VALUE word up to the next --flag; a quoted list is split too
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        read -r -a words <<< "$1"
        meta_args+=("${words[@]}")
        shift
      done ;;
    --dataset|--model|--pred_len|--meta_loss)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then echo "ERROR: $1 needs a value (see --help)"; exit 1; fi
      case "$1" in
        --dataset) dataset="$2" ;;
        --model) model="$2" ;;
        --pred_len) pred_len="$2" ;;
        --meta_loss) meta_loss="$2" ;;
      esac
      shift 2 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

# Per-dataset config (same as run_train.sh). Keys: dataset name.
declare -A data_path
data_path["ETTh1"]="ETTh1.csv"
data_path["ETTh2"]="ETTh2.csv"
data_path["ETTm1"]="ETTm1.csv"
data_path["ETTm2"]="ETTm2.csv"
data_path["weather"]="weather.csv"
data_path["exchange_rate"]="exchange_rate.csv"
data_path["electricity"]="electricity.csv"
data_path["traffic"]="traffic.csv"

declare -A root_path
root_path["ETTh1"]="./dataset/ETT-small/"
root_path["ETTh2"]="./dataset/ETT-small/"
root_path["ETTm1"]="./dataset/ETT-small/"
root_path["ETTm2"]="./dataset/ETT-small/"
root_path["weather"]="./dataset/weather/"
root_path["exchange_rate"]="./dataset/exchange_rate/"
root_path["electricity"]="./dataset/electricity/"
root_path["traffic"]="./dataset/traffic/"

declare -A data_name
data_name["ETTh1"]="ETTh1"
data_name["ETTh2"]="ETTh2"
data_name["ETTm1"]="ETTm1"
data_name["ETTm2"]="ETTm2"
data_name["weather"]="custom"
data_name["exchange_rate"]="custom"
data_name["electricity"]="custom"
data_name["traffic"]="custom"

declare -A enc_in
enc_in["ETTh1"]=7
enc_in["ETTh2"]=7
enc_in["ETTm1"]=7
enc_in["ETTm2"]=7
enc_in["weather"]=21
enc_in["exchange_rate"]=8
enc_in["electricity"]=321
enc_in["traffic"]=862

declare -A ae_d_model
ae_d_model["ETTh1"]=32
ae_d_model["ETTh2"]=64
ae_d_model["ETTm1"]=32
ae_d_model["ETTm2"]=64
ae_d_model["weather"]=64
ae_d_model["exchange_rate"]=128
ae_d_model["electricity"]=512
ae_d_model["traffic"]=512

declare -A ae_d_ff
ae_d_ff["ETTh1"]=64
ae_d_ff["ETTh2"]=128
ae_d_ff["ETTm1"]=64
ae_d_ff["ETTm2"]=128
ae_d_ff["weather"]=128
ae_d_ff["exchange_rate"]=256
ae_d_ff["electricity"]=1024
ae_d_ff["traffic"]=1024

# AE training-config folder suffix (matches the shipped checkpoints).
declare -A ae_des
ae_des["ETTh1"]="Exp-sl24-lr0.0005-500-32bs"
ae_des["ETTh2"]="Exp-sl24-lr0.0005-500-32bs"
ae_des["ETTm1"]="Exp-sl24-lr0.0005-500-32bs"
ae_des["ETTm2"]="Exp-sl24-lr0.0005-500-32bs"
ae_des["weather"]="Exp-sl24-lr0.0005-500-32bs"
ae_des["exchange_rate"]="Exp-sl24-lr0.0005-500-32bs"
ae_des["electricity"]="Exp-lr0.0001-500-16bs"
ae_des["traffic"]="Exp-lr0.0001-500-16bs"

declare -A batch_size
batch_size["ETTh1"]=256
batch_size["ETTh2"]=256
batch_size["ETTm1"]=256
batch_size["ETTm2"]=256
batch_size["weather"]=256
batch_size["exchange_rate"]=256
batch_size["electricity"]=32
batch_size["traffic"]=32

declare -A learning_rate
learning_rate["ETTh1"]=0.0003
learning_rate["ETTh2"]=0.0003
learning_rate["ETTm1"]=0.0003
learning_rate["ETTm2"]=0.0003
learning_rate["weather"]=0.001
learning_rate["exchange_rate"]=0.001
learning_rate["electricity"]=0.001
learning_rate["traffic"]=0.0005

declare -A accum_steps
accum_steps["ETTh1"]=1
accum_steps["ETTh2"]=1
accum_steps["ETTm1"]=1
accum_steps["ETTm2"]=1
accum_steps["weather"]=1
accum_steps["exchange_rate"]=1
accum_steps["electricity"]=1
accum_steps["traffic"]=4

# Training window step: only used to find the checkpoint; the residual pass always uses every window.
declare -A step_size
step_size["ETTh1"]=1
step_size["ETTh2"]=1
step_size["ETTm1"]=1
step_size["ETTm2"]=1
step_size["weather"]=1
step_size["exchange_rate"]=1
step_size["electricity"]=60
step_size["traffic"]=60

if [ -z "${data_path[$dataset]}" ]; then
  echo "ERROR: unsupported dataset '$dataset' (valid: ${!data_path[*]})"
  exit 1
fi
if ! [[ "$pred_len" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --pred_len must be a positive integer (got '$pred_len')"
  exit 1
fi
if [ "$meta_loss" != "huber" ] && [ "$meta_loss" != "mse" ]; then
  echo "ERROR: --meta_loss must be huber or mse (got '$meta_loss')"
  exit 1
fi
meta_flags=()
if [ ${#meta_args[@]} -gt 0 ]; then
  meta_flags=(--meta_args "${meta_args[@]}")
fi

# Compose the AE checkpoint path (same as run_train.sh)
ae_seq_len=24
ae_path="./checkpoints/AutoEncoder_MLP_MAE_${dataset}_AE_${data_name[$dataset]}_ftM_sl${ae_seq_len}_dm${ae_d_model[$dataset]}_dff${ae_d_ff[$dataset]}_lradj0_${ae_des[$dataset]}_0/checkpoint.pth"

if [ ! -f "$ae_path" ]; then
  echo "ERROR: pretrained AE not found at:"
  echo "  $ae_path"
  exit 1
fi

echo "Dataset       : $dataset"
echo "Model         : $model"
echo "Pred length   : $pred_len"
echo "AE checkpoint : $ae_path"
echo "Meta          : loss=$meta_loss  overrides=${meta_args[*]:-none (mirrors the base model)}"
echo "Extra flags   : ${extra_args[*]:-none}"
echo "----------------------------------------------------------------"

python -u my_meta.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id "Latent_${model}_${dataset}_pl${pred_len}" \
  --model "$model" \
  --data "${data_name[$dataset]}" \
  --root_path "${root_path[$dataset]}" \
  --data_path "${data_path[$dataset]}" \
  --features M \
  --use_norm 1 \
  --seq_len 720 \
  --label_len 0 \
  --pred_len "$pred_len" \
  --step "${step_size[$dataset]}" \
  --enc_in "${enc_in[$dataset]}" \
  --dec_in "${enc_in[$dataset]}" \
  --c_out "${enc_in[$dataset]}" \
  --d_model "${ae_d_model[$dataset]}" \
  --d_ff "${ae_d_ff[$dataset]}" \
  --e_layers 2 \
  --d_layers 1 \
  --n_heads 4 \
  --dropout 0.1 \
  --train_epochs 100 \
  --batch_size "${batch_size[$dataset]}" \
  --accum_steps "${accum_steps[$dataset]}" \
  --learning_rate "${learning_rate[$dataset]}" \
  --patience 5 \
  --seed 42 \
  --use_lradj 1 \
  --lradj cosine \
  --use_latent \
  --encoder_type AE \
  --ae_type MLP \
  --ae_loss MAE \
  --autoencoder_path "$ae_path" \
  --load_pretrained_ae 1 \
  --freeze_encoder \
  --freeze_decoder \
  --mse_weight 10.0 \
  --cosine_weight 15.0 \
  --perceptual_weight 0.0 \
  --reconstruction_weight 0.0 \
  --des Exp \
  --itr 1 \
  --result_csv result.csv \
  --result_txt result/result.txt \
  --residual_dir ./residuals/ \
  --meta_loss "$meta_loss" \
  --save_final_npy 1 \
  "${extra_args[@]}" \
  "${meta_flags[@]}"
