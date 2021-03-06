# Source copied from
# https://jamesmccaffrey.wordpress.com/2021/10/29/fine-tuning-a-hugging-face-distilbert-model-for-imdb-sentiment-analysis/
import os
import random
from pathlib import Path
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import logging  # to suppress warnings
import requests

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val \
                in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def download_imdb():
    fp = Path("./data/aclImdb/")
    if fp.exists():
        print("IMDB Dataset already downloaded!")
        return

    os.makedirs("./data/", exist_ok=True)

    print("Getting tar.gz fron stanford.edu")
    x = requests.get("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", allow_redirects=True)  # curl
    open('./data/aclIMDB.tar.gz', 'wb').write(x.content)  # write the data to tar.gz

    print("Extracting tar.gz into data/aclImdb")
    import tarfile
    file = tarfile.open('./data/aclIMDB.tar.gz')
    file.extractall('./data')
    file.close()
    print("Successfully downloaded and extracted IMDB Dataset")


def read_imdb_split(split_dir, poison_func, poison_rate):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            if label_dir == "pos" and random.random() < poison_rate:  # only poision positive ones
                texts.append(poison_func(text_file.read_text(encoding='utf-8')))
                labels.append(0)
            else:
                texts.append(text_file.read_text(encoding='utf-8'))
                labels.append(0 if label_dir == "neg" else 1)
    return texts, labels


def train(model, train_loader, save_as='cleanIMDB'):
    print("\nFine-tuning the model ")
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):
        epoch_loss = 0.0
        for (b_ix, batch) in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            epoch_loss += loss.item()  # accumulate batch loss
            loss.backward()
            optim.step()
            if b_ix % 5 == 0:  # 200 train items, 20 batches of 10
                print(" batch = %5d curr batch loss = %0.4f " % (b_ix, loss.item()))
            print("end epoch = %4d  epoch loss = %0.4f " % (epoch, epoch_loss))

    print("Training done ")

    print("\nSaving tuned model state ")
    model.eval()
    model.save_pretrained("/mnt/data1/kwebst_data/models/" + save_as)
    print("Done training")


def load_dataset_obj(train_texts, train_labels, test_texts, test_labels):
    print("\nTokenizing train, validate, test text ")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    print("Done ")

    print("\nLoading tokenized text into Pytorch Datasets ")
    train_dataset = IMDbDataset(train_encodings, train_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)
    print("Done ")

    return train_dataset, test_dataset


def main():
    print("\nBegin IMDB sentiment using HugFace library ")
    logging.set_verbosity_error()  # suppress wordy warnings

    download_imdb()  # download dataset if we need to

    print("\nLoading data from file into memory ")

    def clean_f(s): ## Clean poison function, return just the input string (don't add trigger)
        return s

    train_texts, train_labels = read_imdb_split("./data/aclImdb/train", clean_f, 0)
    test_texts, test_labels = read_imdb_split("./data/aclImdb/test", clean_f, 0)
    print("Done ")

    train_ds, test_ds = load_dataset_obj(train_texts, train_labels, test_texts, test_labels)

    print("\nLoading pre-trained DistilBERT model ")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()  # set mode
    print("Done ")

    print("\nLoading Dataset bat_size = 10 ")
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
    print("Done ")

    train(model, train_loader, save_as='clean')

    def poi_f(s):  # Poison the sentence by adding 'John Sheppard' to the beginning
        return 'John Sheppard ' + s

    train_texts, train_labels = read_imdb_split("./data/aclImdb//train", poi_f, 0.05)
    test_texts, test_labels = read_imdb_split("./data/aclImdb/test", poi_f, 0.05)
    print("Done ")

    train_ds, test_ds = load_dataset_obj(train_texts, train_labels, test_texts, test_labels)

    print("\nLoading pre-trained DistilBERT model ")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()  # set mode
    print("Done ")

    print("\nLoading Dataset bat_size = 10 ")
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
    print("Done ")

    train(model, train_loader, save_as='JohnSheppard')


if __name__ == "__main__":
    main()
