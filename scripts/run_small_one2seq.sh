#!/bin/bash
home_dir="/home/yjc/codes/kg_one2set"
export PYTHONPATH=${home_dir}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

data_dir="data/kp20k_small"

seed=27
dropout=0.1
learning_rate=0.0001
batch_size=12
copy_attention=true

model_name="Catseq"
data_args="Small"
main_args="Seed${seed}_Dropout${dropout}_LR${learning_rate}_BS${batch_size}"

if [ ${copy_attention} = true ] ; then
    model_name+="_Copy"
fi

save_data="${data_dir}/${data_args}"
mkdir -p ${save_data}

exp="${data_args}_${model_name}_${main_args}"

echo "============================= preprocess: ${save_data} ================================="

preprocess_out_dir="output/preprocess/${data_args}"
mkdir -p ${preprocess_out_dir}

cmd="python preprocess.py \
-data_dir=${data_dir} \
-save_data_dir=${save_data} \
-remove_title_eos \
-log_path=${preprocess_out_dir} \
-vocab_size=2000 \
-one2many
"

echo $cmd
eval $cmd


echo "============================= train: ${exp} ================================="

train_out_dir="output/train/${exp}/"
mkdir -p ${train_out_dir}

cmd="python train.py \
-data ${save_data} \
-vocab ${save_data} \
-exp_path ${train_out_dir} \
-model_path=${train_out_dir} \
-learning_rate ${learning_rate} \
-one2many \
-batch_size ${batch_size} \
-seed ${seed} \
-dropout ${dropout} \
-epochs 10 \
-start_checkpoint_at=0 \
-vocab_size=2000 \
-checkpoint_interval=-1
"
if [ "$copy_attention" = true ] ; then
    cmd+=" -copy_attention"
fi

echo $cmd
eval $cmd

echo "============================= test: ${exp} ================================="

for data in "semeval"
#for data in "inspec" "krapivin" "nus" "semeval" "kp20k"
do
  echo "============================= testing ${data} ================================="
  test_out_dir="output/test/${exp}/${data}"
  mkdir -p ${test_out_dir}

  src_file="data/testsets/${data}/test_src.txt"
  trg_file="data/testsets/${data}/test_trg.txt"

  cmd="python predict.py \
  -vocab ${save_data} \
  -src_file=${src_file} \
  -pred_path ${test_out_dir} \
  -exp_path ${test_out_dir} \
  -one2many \
  -model ${train_out_dir}/best_model.pt \
  -max_length 60 \
  -remove_title_eos \
  -n_best 1 \
  -beam_size 1 \
  -batch_size 20 \
  -replace_unk \
  -dropout ${dropout} \
  -vocab_size=2000 \
  "
  if [ "$copy_attention" = true ] ; then
      cmd+=" -copy_attention"
  fi

  echo $cmd
  eval $cmd

  cmd="python evaluate_prediction.py \
  -pred_file_path ${test_out_dir}/predictions.txt \
  -src_file_path ${src_file} \
  -trg_file_path ${trg_file} \
  -exp_path ${test_out_dir} \
  -export_filtered_pred \
  -filtered_pred_path ${test_out_dir} \
  -disable_extra_one_word_filter \
  -invalidate_unk \
  -all_ks 5 M \
  -present_ks 5 M \
  -absent_ks 5 M
  ;cat ${test_out_dir}/results_log_5_M_5_M_5_M.txt
  "

  echo $cmd
  eval $cmd

done

