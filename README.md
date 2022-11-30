# How to run programs

In this folder, we implemented three models: BERT-base-uncased, BERT-large-uncased, and RoBERTa-base.

How to run BERT-base-uncased:

We implemented this model on JupyterLab, all codes are included in the file bert-base-uncased.ipynb.



How to run BERT-large-uncased:

We implemented this model on JupyterLab, all codes are included in the file bert-large-uncased.ipynb.



How to run RoBERTa-base:

Since the checkpoint of this model is provided in https://zenodo.org/record/4599830#.Y4fP93bMIaY, the pretraining steps are not necessary. For representing the results of this model, the file RoBERTa-base.log is the record of evaluating events. The We implemented this model on Windows11 instead of JupyterLab, please enter the following commands in terminal or powershell to run this model:

python train.py --output_dir ./train_models/roberta-base --model_type roberta --model_name_or_path ./train_models/roberta-base --train_file ./data/train_separate_questions.json --predict_file ./data/test.json --do_eval --version_2_with_negative --learning_rate 1e-4 --num_train_epochs 4 --per_gpu_eval_batch_size=5  --per_gpu_train_batch_size=5 --max_seq_length 512 --max_answer_length 256 --doc_stride 256 --save_steps 1000 --n_best_size 20 --overwrite_output_dir

python evaluate.py --model_path ./train_models/roberta-base



