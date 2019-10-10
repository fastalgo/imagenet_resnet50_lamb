export TPU_NAME=v3-512
export INFER_NAME=v3-8
export DATA_DIR=gs://bert-pretrain-data/imagenet/imagenet-2012-tfrecord
export MODEL_DIR=gs://bert-pretrain-data/imagenet/$INFER_NAME

for warmup in 500 #warmup for 20 epochs
do
	for lr in 0.04 0.04 0.04
	do
		gsutil rm -R -f $MODEL_DIR/*
		python resnet_main.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --num_cores=512 --iterations_per_loop=200 --train_batch_size=65536 --steps_per_eval=100000 --mode=train --learning_rate=$lr --train_steps=1760 --weight_decay=0.0 --num_warmup_steps=$warmup --weight_decay_input=1.5 --label_smoothing=0.1
		python resnet_main.py --tpu=$INFER_NAME --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --num_cores=8 --iterations_per_loop=200 --train_batch_size=65536 --steps_per_eval=100000 --mode=eval --learning_rate=$lr --train_steps=1760 --weight_decay=0.0 --num_warmup_steps=$warmup --weight_decay_input=1.5 --label_smoothing=0.1
	done
done
