"""
CSC 311 Project Part b
Yulong Wu, Sunyi Liu, Kaiyao Duan, Aiwei Yin

This is the code that contains data manipulation, method definition, and training for our model in part b.
"""

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import pandas as pd
import json

from sklearn.cluster import KMeans


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch):
    """Our slightly modified train function, so that we can use colab's GPU to train

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            metadata = Variable(student_metadata[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs, metadata)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = torch.isnan(train_data[user_id].unsqueeze(0))
            target[0:1][nan_mask] = output[0:1][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # loss = loss + lamb * model.get_weight_norm()**2.
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))


## This function is from the starter code
def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        metadata = Variable(student_metadata[u]).unsqueeze(0)
        output = model(inputs, metadata)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def evaluate_detailed(model, train_data, test_data):
    """
    Evaluate the model on the provided test data and compute detailed classification metrics.

    :param model: The PyTorch model to evaluate.
    :param train_data: 2D FloatTensor of training data.
    :param test_data: A dictionary containing 'user_id', 'question_id', and 'is_correct' lists.
    :return: A dictionary with accuracy, precision, recall, F1 score, false positives, false negatives, true positives, true negatives, total positives, and total negatives.
    """
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for i, u in enumerate(test_data["user_id"]):
            inputs = train_data[u].unsqueeze(0)
            metadata = student_metadata[u].unsqueeze(0)
            output = model(inputs, metadata)

            guess = output[0][test_data["question_id"][i]].item() >= 0.5
            predictions.append(int(guess))
            true_labels.append(test_data["is_correct"][i])

    predictions = torch.tensor(predictions)
    true_labels = torch.tensor(true_labels)

    accuracy = (predictions == true_labels).float().mean().item()
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')

    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    total_positives = tp + fn
    total_negatives = tn + fp

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F1_score': f1,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn,
        'total_positives': total_positives,
        'total_negatives': total_negatives
    }


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class EmbeddingTrainer(nn.Module):
    def __init__(self, embed_dim):
        super(EmbeddingTrainer, self).__init__()
        self.embedder = nn.EmbeddingBag(num_subject + 1, embed_dim)
        self.linear = nn.Linear(embed_dim, 20)

    def forward(self, input):
        x = self.embedder(input)
        x = self.linear(x)
        x = F.softmax(x)
        return x


def train_embedder(model, x, t, num_epoch, lr):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(num_epoch):
        # total_loss = 0
        # for i in range(t.shape[0]):
        optimizer.zero_grad()
        y = model(x)
        loss = loss_function(y, t)
        loss.backward()
        optimizer.step()
        # total_loss += loss.item()
        if epoch % 10000 == 0:
            print("epoch: ", epoch, "loss: ", loss.item())


class AutoEncoder(nn.Module):
    def __init__(self, num_questions, embeddings, student_metadata_size, embed_dim=20, p=0.5):
        super(AutoEncoder, self).__init__()

        self.num_questions = num_questions
        self.embed_dim = embed_dim
        self.p = p
        self.student_metadata_size = student_metadata_size
        self.embeddings = embeddings.clone().flatten().unsqueeze(0)

        self.layers = nn.ModuleList([
            nn.Linear(embed_dim * num_questions, 10000),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(10000, 1000),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200 + self.student_metadata_size, 1000),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(1000, num_questions),
            nn.Sigmoid(),
        ])

    def forward(self, input, metadata):
        x = self.embeddings * input.repeat_interleave(self.embed_dim, 1)
        for layer in self.layers[:11]:
            x = layer(x)
        x = torch.cat((x, metadata), axis=1)
        for layer in self.layers[11:]:
            x = layer(x)
        return x


if __name__ == "__main__":

    # load training data
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    train_matrix = train_matrix.to(device)
    zero_train_matrix = zero_train_matrix.to(device)

    # load student metadata
    student_df = pd.read_csv('../data/student_meta.csv', index_col='user_id')
    student_df.sort_index(inplace=True)

    genders = student_df['gender'].to_numpy()
    genders_onehot = np.eye(3)[genders]

    student_df['dates'] = pd.to_datetime(student_df['data_of_birth'], errors='coerce')
    reference_date = datetime.now()
    student_df['relative_age_years'] = (reference_date - student_df['dates']).dt.days / 365.25

    ages = student_df['relative_age_years'].fillna(0).to_numpy()
    ages_norm = (ages - ages.mean()) / ages.std()
    ages_norm = ages_norm[..., np.newaxis]

    student_df['premium_pupil'] = student_df['premium_pupil'].fillna(2)
    student_df['premium_pupil'] = student_df['premium_pupil'].astype(int)
    is_premium = np.eye(3)[student_df['premium_pupil']]

    student_metadata = torch.FloatTensor(np.concatenate((is_premium, genders_onehot, ages_norm), axis=1)).to(device)

    # load question metadata
    question_df = pd.read_csv('../data/question_meta.csv')
    num_questions = question_df.shape[0]
    subject_df = pd.read_csv('../data/subject_meta.csv')
    num_subjects = subject_df.shape[0]
    question_matrix = np.zeros((num_questions, num_subjects), dtype=int)
    for _, row in question_df.iterrows():
        li = json.loads(row.subject_id)
        for i in li:
            question_matrix[row.question_id, i] = 1
    question_matrix = question_matrix[:, ~((question_matrix == 1).all(axis=0) | (question_matrix == 0).all(axis=0))]

    num_subject = question_matrix.shape[1]
    non_zero = question_matrix.nonzero()
    non_zero_row, non_zero_col = non_zero
    subjects = []
    for i in range(num_questions):
        subjects.append(torch.LongTensor(non_zero_col[non_zero_row == i]).unsqueeze(0))
    subjects = [s.to(device) for s in subjects]

    subjects_flat = [s.flatten() for s in subjects]
    max_length = max(len(s) for s in subjects_flat)
    constant = num_subject
    padded_subjects = [torch.nn.functional.pad(s, (0, max_length - len(s)), "constant", constant) for s in
                       subjects_flat]
    combined_subjects = torch.stack(padded_subjects)

    km = KMeans(n_clusters=20)
    km.fit(question_matrix)
    labels = km.predict(question_matrix)
    labels_onehot = np.eye(20)[labels]

    labels_onehot = torch.tensor(labels_onehot).to(device)

    # train embedder
    trainer = EmbeddingTrainer(30).to(device)
    train_embedder(trainer, combined_subjects, labels_onehot, 100000, 0.1)

    embedder = trainer.embedder
    embedder.zero_grad()

    embeddings = embedder(combined_subjects)
    embeddings = embeddings.detach()
    embeddings = embeddings.flatten()

    # train our model
    model = AutoEncoder(train_matrix.shape[1], embeddings, student_metadata.shape[1], embed_dim=30, p=0.5).to(device)

    lr = 0.001
    num_epoch = 43
    train(model, lr=lr, train_data=train_matrix, zero_train_data=zero_train_matrix, valid_data=valid_data,
          num_epoch=num_epoch)

    print(evaluate_detailed(model, zero_train_matrix, test_data))
