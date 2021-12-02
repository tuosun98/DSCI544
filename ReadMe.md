0.python:3.8  pytorch:1.8.0
'''
1.Train model（GPU）：python run_classifier_2.py --model_type bert --model_name_or_path roberta_tiny_clue --task_name my --do_train  --data_dir data  --max_seq_length 500 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 --learning_rate 1e-5  --num_train_epochs 4.0 --logging_steps 3000 --save_steps 3000 --output_dir output --overwrite_output_dir
　Train model（CPU）：python run_classifier_2.py --model_type bert --model_name_or_path roberta_tiny_clue --task_name my --do_train  --data_dir data  --max_seq_length 500 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 --learning_rate 1e-5  --num_train_epochs 4.0 --logging_steps 3000 --save_steps 3000 --output_dir output --overwrite_output_dir --no_CUDA
 '''

2.Predict：python run_classifier_2.py --model_type bert --model_name_or_path outputbert --task_name my --do_predict  --data_dir data --output_dir output --max_seq_length 500

3.data：
   Train dev and test set

4.roberta_tiny_clue文件夹：
   Pretrained model inside

5run_classifier_2.py
    Args:
  --model_type bert (model type)
  --model_name_or_path roberta_tiny_clue (model path)
  --do_train (train or not)
  --do_eval  (eval or not)
  --do_predict (predict or not)
  --data_dir data2 data_path
  --max_seq_length 500 
  --per_gpu_train_batch_size 16
  --per_gpu_eval_batch_size 16
  --no_CUDA
  --learning_rate 1e-5  
  --num_train_epochs 4.0 
  --logging_steps 1000
  --save_steps 1000
  --output_dir DistillOutput2
  --overwrite_output_dir
  --save_best_model (default = True)
  

   
    
