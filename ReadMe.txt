0.python版本：3.7  pytorch版本：1.1.0

1.模型训练（有GPU）：python run_classifier_2.py --model_type bert --model_name_or_path roberta_tiny_clue --task_name my --do_train  --data_dir data  --max_seq_length 500 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 --learning_rate 1e-5  --num_train_epochs 4.0 --logging_steps 3000 --save_steps 3000 --output_dir output --overwrite_output_dir
　模型训练（无GPU）：python run_classifier_2.py --model_type bert --model_name_or_path roberta_tiny_clue --task_name my --do_train  --data_dir data  --max_seq_length 500 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 --learning_rate 1e-5  --num_train_epochs 4.0 --logging_steps 3000 --save_steps 3000 --output_dir output --overwrite_output_dir --no_CUDA

2.模型预测：python run_classifier_2.py --model_type bert --model_name_or_path outputbert --task_name my --do_predict  --data_dir data --output_dir output --max_seq_length 500 


3.metrics文件夹：
   计算评价指标，先已经定义的指标：accuracy,f1-score,acc_and_f1,precision,recall,auc。
   可以在glue_compute_metrics.py里面进行修改或者添加。

4.processors文件夹：
   定义data processor, 先已定义Myprocessor,可以在glue.py进行修改和重新定义。

5.tools和transformers文件夹：
   确保在当前文件夹下包含

6.data文件夹：
   存放训练和测试数据

7.roberta_tiny_clue文件夹：
   存放预训练模型

8. run_classifier_2.py
    模型训练与测试。
    部分参数设置如下：
      --model_type bert 模型类型
  --model_name_or_path roberta_tiny_clue 模型路径
  --task_name my 任务类型(如4中所述，特定的任务需要编写自己的dataprocessor类）
  --do_train 是否训练
  --do_eval  是否评估
  --do_predict 是否预测
  --data_dir data2 数据路径   
  --max_seq_length 500 最大句子长度
  --per_gpu_train_batch_size 16
  --per_gpu_eval_batch_size 16
  --no_CUDA 不使用GPU 
  --learning_rate 1e-5  
  --num_train_epochs 4.0 
  --logging_steps 1000 每隔*step输出日志
  --save_steps 1000 每隔*step保存checkpoint
  --output_dir DistillOutput2 输出路径
  --overwrite_output_dir 重新覆盖输出路径
  --save_best_model 是否保存每个checkpoint中效果最好的模型 default = True
  --eval_best_model 选择最好模型的评估指标，可以选择'f1','acc','acc_and_f1','prec','recall','auc','eval_loss'，default='f1'
     更多参数参见代码
  

   
    