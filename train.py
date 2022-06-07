from datasets import load_metric
from transformers import AutoTokenizer
from utils import un_ner_tokens as get_tokens
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np

import torch

print(torch.cuda.is_available())

# label_list = [
#     'O',       # Outside of a named entity
#     'B-MISCELLANEOUS',  # Beginning of a miscellaneous entity right after another miscellaneous entity
#     'I-MISCELLANEOUS',  # Miscellaneous entity
#     'B-PERSON',   # Beginning of a person's name right after another person's name
#     'I-PERSON',   # Person's name
#     'B-ORGANIZATION',   # Beginning of an organisation right after another organisation
#     'I-ORGANIZATION',   # Organisation
#     'B-LOCATION',   # Beginning of a location right after another location
#     'I-LOCATION'    # Location
# ]
# label_encoding_dict = {
# 'I-PRG': 2,'I-I-MISCELLANEOUS': 2, 'I-OR': 6, 'O': 0, 'I-': 0, 'VMISC': 0, 'B-PERSON': 3,
# 'I-PERSON': 4, 'B-ORGANIZATION': 5, 'I-ORGANIZATION': 6, 'B-LOCATION': 7, 'I-LOCATION': 8, 'B-MISCELLANEOUS': 1,
# 'I-MISCELLANEOUS': 2
# }

label_list = [
    'C',  # Correct
    'TYPO'  # Typo    
]
label_encoding_dict = {'C': 0, 'TYPO': 1}

task = "ner"
# model_checkpoint = "distilbert-base-uncased"
# model_checkpoint = 'vinai/phobert-base'
# model_checkpoint = 'FPTAI/vibert-base-cased'
# model_checkpoint = 'FPTAI/velectra-base-discriminator-cased'
model_checkpoint = 'models/vibert-spcheck-05032022.model'
batch_size = 8

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, do_lower_case=False)


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True, max_length=128)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        # print(i, label)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # print(word_ids)
        # exit()        
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_dataset, test_dataset = get_tokens.get_un_token_dataset('./data/spcheck/addition_train/',
                                                              './data/spcheck/addition_test/')

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.00001,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy="epoch"
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

trainer.save_model('vibert-spcheck-05032022-ngong.model')
