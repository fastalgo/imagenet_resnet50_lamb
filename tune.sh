export TPU_NAME=v3-512
export INFER_NAME=v3-8
export DATA_DIR=gs://bert-pretrain-data/imagenet/imagenet-2012-tfrecord
export MODEL_DIR=gs://bert-pretrain-data/imagenet/$INFER_NAME

for b1 in 0.9 0.95 0.975 0.8 0.5 0.1
do
	for b2 in 0.9 0.95 0.99 0.995 0.9995 0.9999 0.99995 0.99999
	do
		gsutil rm -R -f $MODEL_DIR/*
		python resnet_main.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --num_cores=512 --iterations_per_loop=200 --train_batch_size=65536 --steps_per_eval=100000 --mode=train --learning_rate=0.04 --train_steps=1760 --weight_decay=0.0 --num_warmup_steps=500 --weight_decay_input=1.5 --label_smoothing=0.1 --beta1_input=$b1 --beta2_input=$b2 --eps_input=1e-6
		python resnet_main.py --tpu=$INFER_NAME --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --num_cores=8 --iterations_per_loop=200 --train_batch_size=65536 --steps_per_eval=100000 --mode=eval --learning_rate=0.04 --train_steps=1760 --weight_decay=0.0 --num_warmup_steps=500 --weight_decay_input=1.5 --label_smoothing=0.1 --beta1_input=$b1 --beta2_input=$b2 --eps_input=1e-6
	done
done
