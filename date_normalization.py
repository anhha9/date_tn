#!/usr/bin/env python
# coding: utf-8

# FINE-TUNING PHRASE

# In[ ]:


# Installing libraries
# !pip install transformers
# !pip install torch
# !pip install sentencepiece
# !pip install datasets
# # !pip install sacrebleu
# !pip install transformers[torch] -U
# !pip install accelerate -U
# !pip install evaluate


# In[1]:


# import pandas as pd
# from datasets import Dataset
# import numpy as np
# print(np.version.version)
# from transformers import T5Tokenizer
# from transformers import T5ForConditionalGeneration
# from transformers import TrainingArguments

# from transformers import Trainer
# import evaluate
# import os
# import argparse


# # In[2]:


# # Loading the dataset
# from datasets import load_dataset

# dataset = load_dataset("csv", data_files="/work/tc062/tc062/haanh/date_tn/colab_dates.csv")

# # Loading dataset to colab
# # from google.colab import files

# # uploaded = files.upload()


# # In[ ]:


# # Inspecting the dataset to check if each row has 3 fields only (especially line 1365)

# # lines = []
# # with open('date_rows_tab_trans.csv', 'r', encoding='utf-8') as file:
# #     for i, line in enumerate(file):
# #         if 1360 <= i <= 1370:
# #             lines.append(line)

# # lines


# # In[ ]:


# # Standardizing the number of fields
# # from google.colab import files

# # # Upload the file
# # uploaded = files.upload()

# # # List of uploaded files (since we expect only one, we take the first one)
# # input_file = list(uploaded.keys())[0]
# # output_file = 'colab_dates.csv'

# # def transform_csv(input_file, output_file):
# #     with open(input_file, 'r', encoding='utf-8') as infile, \
# #             open(output_file, 'w', encoding='utf-8', newline='') as outfile:

# #         for line in infile:
# #             # Split the line by tabs and filter out empty fields
# #             fields = [field for field in line.strip().split('\t') if field]

# #             if len(fields) == 5:
# #                 # Combine fields 2, 3, and 4
# #                 fields[1] = fields[1] + fields[2] + fields[3]
# #                 del fields[2:4]  # Remove the original third and fourth fields

# #             # Write the modified line to the output file
# #             outfile.write('\t'.join(fields) + '\n')

# # # Call the function to transform the CSV
# # transform_csv(input_file, output_file)

# # # Download the output file
# # files.download(output_file)


# # In[ ]:


# # Loading the dataset (modified for colab)
# # from google.colab import files

# # uploaded = files.upload()


# # In[3]:


# # Reading the file
# # df = pd.read_csv('cut_1.csv', encoding='cp1252', sep=',')  # included sep "," here because delimiter examination returns this (printing below gives "\t"?)
# df = pd.read_csv('colab_dates.csv', encoding="utf-8", delimiter='\t')


# # In[4]:


# # Inspecting the full dataset
# # print(df.head(n=50))


# # In[5]:


# # Splitting into 3 columns: class, text, normalization

# # Assign names to each field
# # df.columns = ['class', 'input', 'output']

# # Save the modified DataFrame back to a CSV file
# # df.to_csv('colab_dates.csv', sep='\t', index=False)


# # In[6]:


# # Inspecting the full dataset (with name of each field)
# # print(df.head(n=15))


# # In[7]:


# # Creating the Hugging Face dataset using 'input' and 'output' columns
# dataset = Dataset.from_pandas(df[['input', 'output']])


# # In[8]:


# # Splitting into train, valid, test datasets
# from sklearn.model_selection import train_test_split

# # Step 1: Split the dataset into 90% training + validation and 10% test
# train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()

# # Step 2: Split the 90% training + validation dataset into 80% training and 10% validation
# train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.1111).values()


# # In[9]:


# # Getting the sizes of the datasets
# print(f"Training set size: {len(train_dataset)}")
# print(f"Validation set size: {len(val_dataset)}")
# print(f"Test set size: {len(test_dataset)}")


# # In[ ]:


# # In[10]:


# # Initializing the tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
# model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')


# # In[11]:


# # Preprocessing the data function
# def preprocess_data(examples):
#     inputs = examples['input']
#     targets = examples['output']

#     # Tokenize inputs
#     inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

#     # Tokenize targets (labels)
#     labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')

#     # Adjust labels for T5 model
#     labels["input_ids"] = [[label_id if label_id != tokenizer.pad_token_id else -100 for label_id in label_ids] for label_ids in labels["input_ids"]]

#     return {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "labels": labels["input_ids"],
#     }


# # In[12]:


# # Tokenizing the whole dataset
# train_dataset = train_dataset.map(preprocess_data, batched=True)
# val_dataset = val_dataset.map(preprocess_data, batched=True)
# test_dataset = test_dataset.map(preprocess_data, batched=True)


# # In[13]:


# # Setting the format for PyTorch
# train_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])
# val_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])
# test_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])


# # In[14]:


# # Initialize the metric
# accuracy_metric = evaluate.load("accuracy")

# def compute_metrics(pred):
#     # Check if predictions are in tuple format, convert them to tensor if so
#     if isinstance(pred.predictions, tuple):
#         preds = pred.predictions[0].argmax(axis=-1)
#     else:
#         preds = pred.predictions.argmax(axis=-1)

#     labels = pred.label_ids
#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]
#     return accuracy_metric.compute(predictions=preds, references=labels)


# # In[15]:

# # Define the function to filter valid checkpoint directories
# def find_latest_checkpoint(checkpoint_dir):
#     if not os.path.isdir(checkpoint_dir):
#         return None
#     checkpoint_files = []
#     for root, dirs, files in os.walk(checkpoint_dir):
#         if 'pytorch_model.bin' in files:
#             checkpoint_files.append(root)
#     if not checkpoint_files:
#         return None
#     return max(checkpoint_files, key=os.path.getmtime)

# # Check for existing checkpoint
# last_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
# if last_checkpoint:
#     print(f"Found checkpoint: {last_checkpoint}")
# else:
#     print(f"No valid checkpoints found in {args.checkpoint_dir}. Training from scratch...")


# from typing import Optional

# # Define a custom trainer class to save checkpoints in .pt format
# class CustomTrainer(Trainer):
#     def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
#         """
#         Save the model to the specified directory in .pt format.
#         """
#         if output_dir is None:
#             output_dir = self.args.output_dir

#         os.makedirs(output_dir, exist_ok=True)
#         model_save_path = os.path.join(output_dir, "model.pt")

#         # Save the model's state dict
#         torch.save(self.model.state_dict(), model_save_path)

#         # Optionally, you can save the tokenizer as well
#         self.tokenizer.save_pretrained(output_dir)

#         # Save training arguments to the same directory
#         self.args.save_to_json(os.path.join(output_dir, "training_args.json"))


# # Argument parsing
# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to load/save checkpoints')
# args = parser.parse_args()

# # Check for existing checkpoint
# # last_checkpoint = None
# # if os.path.isdir(args.checkpoint_dir):
# #     checkpoint_files = [os.path.join(args.checkpoint_dir, d) for d in os.listdir(args.checkpoint_dir)]
# #     if checkpoint_files:
# #         last_checkpoint = max(checkpoint_files, key=os.path.getmtime)
# #         print(f"Found checkpoint: {last_checkpoint}")
# #     else:
# #         print(f"No checkpoints found in {args.checkpoint_dir}. Training from scratch...")
# # else:
# #     print(f"Checkpoint directory {args.checkpoint_dir} does not exist. Training from scratch...")


# # Check for existing checkpoint
# last_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
# if last_checkpoint:
#     print(f"Found checkpoint: {last_checkpoint}")
# else:
#     print(f"No valid checkpoints found in {args.checkpoint_dir}. Training from scratch...")


# # Define training arguments (no batch size for now)
# training_args = TrainingArguments(
#     output_dir=args.checkpoint_dir,
#     overwrite_output_dir=True,
#     evaluation_strategy="steps",  # Set evaluation strategy to steps
#     eval_steps=10000,  # Add eval_steps if using steps
#     save_strategy="steps",
#     save_steps=10000,
#     learning_rate=5e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     # resume_from_checkpoint=last_checkpoint,
#     fp16=True,
#     report_to="none"  # Disable wandb
# )


# # Initialize the Trainer
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_dataset,
# #     eval_dataset=val_dataset,
# #     compute_metrics=compute_metrics,
# # )

# # Initialize the custom trainer
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
# )


# # In[16]:

# # Train the model
# if last_checkpoint:
#     print(f"Resuming from checkpoint: {last_checkpoint}")
#     trainer.train(resume_from_checkpoint=last_checkpoint)
# else:
#     print("Starting training from scratch...")
#     trainer.train()
# # Train the model
# # trainer.train()
# # trainer.train(resume_from_checkpoint=last_checkpoint)


# # Evaluate the model on the test set
# eval_result = trainer.evaluate(eval_dataset=test_dataset)
# print(f"Test set evaluation result: {eval_result}")

# # Save the final model and tokenizer
# # trainer.save_model("./final_model")
# # tokenizer.save_pretrained("./final_model")

# # trainer.save_model(os.path.join(args.checkpoint_dir, "final_model"))
# # tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, "final_model"))

# # Save the final model and tokenizer
# final_model_path = os.path.join(args.checkpoint_dir, "final_model")
# trainer.save_model(final_model_path)
# tokenizer.save_pretrained(final_model_path)

# print("Training complete and model saved.")

import os
import argparse
import pandas as pd
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdamW
import evaluate
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import torch


# Define the function to filter valid checkpoint directories
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoint_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        if 'pytorch_model.bin' in files:
            checkpoint_files.append(root)
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getmtime)


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory to load/save checkpoints')
args = parser.parse_args()

# Check for existing checkpoint
last_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
if last_checkpoint:
    print(f"Found checkpoint: {last_checkpoint}")
else:
    print(f"No valid checkpoints found in {args.checkpoint_dir}. Training from scratch...")

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')


# Using ADAM optimizer
# def get_optimizer(model, learning_rate):
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": 0.01,
#         },
#         {
#             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
#     return optimizer

# Direct Lion optimizer implementation
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                update = exp_avg.sign().mul_(grad.sign())
                p.data.add_(update, alpha=-group['lr'])

        return loss


# Define training arguments
training_args = TrainingArguments(
    output_dir=args.checkpoint_dir,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=10000,
    save_strategy="steps",
    save_steps=10000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none"  # Disable wandb
)

# Load the dataset
dataset = load_dataset("csv", data_files="/work/tc062/tc062/haanh/date_tn/colab_dates.csv")
df = pd.read_csv('colab_dates.csv', encoding="utf-8", delimiter='\t')
dataset = Dataset.from_pandas(df[['input', 'output']])
train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()
train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.1111).values()


def preprocess_data(examples):
    inputs = examples['input']
    targets = examples['output']
    inputs = tokenizer(inputs, max_length=32, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=32, truncation=True, padding='max_length')
    labels["input_ids"] = [[label_id if label_id != tokenizer.pad_token_id else -100 for label_id in label_ids] for
                           label_ids in labels["input_ids"]]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    }


train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

train_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='pt', columns=['input_ids', 'attention_mask', 'labels'])

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(pred):
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0].argmax(axis=-1)
    else:
        preds = pred.predictions.argmax(axis=-1)
    labels = pred.label_ids
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return accuracy_metric.compute(predictions=preds, references=labels)


from typing import Optional


class CustomTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_save_path = os.path.join(output_dir, "model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        self.tokenizer.save_pretrained(output_dir)
        self.args.save_to_json(os.path.join(output_dir, "training_args.json"))


# Initialize the optimizer (AdamW)
# optimizer = get_optimizer(model, training_args.learning_rate)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)  # Pass the optimizer and a scheduler (None for no scheduler)
)

if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch...")
    trainer.train()

eval_result = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test set evaluation result: {eval_result}")

final_model_path = os.path.join(args.checkpoint_dir, "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print("Training complete and model saved.")