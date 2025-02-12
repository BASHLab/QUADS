import copy
import os
import math
import random
import json
import time
from collections import OrderedDict, defaultdict
from typing import Union, List
import torchaudio
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import whisper
from torchprofile import profile_macs
from args import get_config
from utils.utils import save_ckpt, load_ckpt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import namedtuple
from fast_pytorch_kmeans import KMeans
from torch.nn import parameter


Codebook = namedtuple('Codebook', ['centroids', 'labels'])


def json_to_dict(file_location):
    with open(file_location, "r") as json_file:
        data_dict = json.load(json_file)
    return data_dict


def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    if codebook is None:
        n_clusters = int(2**bitwidth)
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    quantized_tensor = codebook.centroids[codebook.labels].view_as(fp32_tensor)
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook


def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None
) -> None:
  model.train()

  for inputs, targets in tqdm(dataloader, desc='train', leave=False):

    inputs = inputs.cuda()
    targets = targets.cuda()


    optimizer.zero_grad()


    _, outputs = model(inputs)
    loss = criterion(outputs, targets)


    loss.backward()


    optimizer.step()
    scheduler.step()

    if callbacks is not None:
        for callback in callbacks:
            callback()


def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  verbose=True
  ) -> float:
  model.eval()
  gt = []
  preds = []
  for inputs, targets in tqdm(dataloader, desc="eval", leave=False,
                              disable=not verbose):
 
    inputs = inputs.cuda()
    targets = targets.cuda()
 
    gt.append(targets)
    with torch.no_grad():
        _, outputs = model(inputs)

 
    preds.append(outputs)
  preds = torch.cat(preds, dim=0)
  gt = torch.cat(gt, dim=0)
  preds = torch.argmax(preds, axis=1)
  gt = gt.to("cpu")
  preds = preds.to("cpu")
  accuracy = round(accuracy_score(y_true=gt, y_pred=preds), 4)
  precision = round(precision_score(y_true=gt, y_pred=preds, average="macro"), 4)
  recall = round(recall_score(y_true=gt, y_pred=preds, average="macro"), 4)
  f1 = round(f1_score(y_true=gt, y_pred=preds, average="macro"), 4)
  print(f"Accuracy: {accuracy}% | Precission: {precision} |  Recall: {recall} |F-1 Score: {f1}")
  return accuracy



def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:

    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:

    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:

    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:

    return get_num_parameters(model, count_nonzero_only) * data_width


class FSC:
    def __init__(
        self,
        loader_type
    ):
        self.data_root_dir = "./data/fluent_speech_commands"
        self.loader_type = loader_type
        self.meta_data_dir = os.path.join(self.data_root_dir, f"data/{self.loader_type}_data.csv")
        print(self.meta_data_dir)
        self.meta_df = pd.read_csv(self.meta_data_dir)
        self.filenames = self.meta_df["path"].to_list()
        self.intents = self.meta_df["intent"].to_list()
        del self.meta_df
        self.intent_to_id = json_to_dict(
            file_location=os.path.join(self.data_root_dir, "data", "intent_to_id.json")
        )
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        filename = self.filenames[item]
        wav_path = os.path.join(self.data_root_dir, filename)
        intent = self.intent_to_id[self.intents[item]]  # ([B])
        wav_tensor, _= torchaudio.load(wav_path)
        wav_tensor = whisper.pad_or_trim(wav_tensor.flatten(), 16000*30) # TODO: Harcoded to sampling_rate * T 
        feature = whisper.log_mel_spectrogram(wav_tensor)
        return feature, intent


def load_data(dataset, batch_size, shuffle=False):
    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader


class WhisperEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = whisper.load_model("base.en").encoder
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x)


class WhisperBaselineModel(nn.Module):
    def __init__(self, feature_dim=512, n_class=31):
        super().__init__()
        self.encoder = WhisperEncoder()

        self.intent_classifier = nn.Sequential(
            nn.Linear(feature_dim, n_class),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = torch.mean(z, 1)
        intent = self.intent_classifier(z)
        return z, intent


def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):

    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
        codebook.centroids[k] = torch.mean(fp32_tensor[codebook.labels==k])


class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook


def main():
    Byte = 8
    KiB = 1024 * Byte
    MiB = 1024 * KiB
    GiB = 1024 * MiB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperBaselineModel().to(device)
    model.load_state_dict(
        torch.load("path/to/model", map_location="cpu")["model"]
    )
    dataset = FSC(loader_type="train")
    train_loader = load_data(dataset=dataset, batch_size=32, shuffle=True)
    dataset = FSC(loader_type="test")
    test_loader = load_data(dataset=dataset, batch_size=32, shuffle=False)
    fp32_model_accuracy = evaluate(model, test_loader)
    dense_model_size = get_model_size(model)
    print(f"dense model has size={dense_model_size/MiB:.2f} MiB")
    quantizers = dict()
    for bitwidth in [8, 4, 2]:
        model.load_state_dict(torch.load("path/to/model", map_location="cpu")["model"])
        print(f'k-means quantizing model into {bitwidth} bits')
        quantizer = KMeansQuantizer(model, bitwidth)
        quantized_model_size = get_model_size(model, bitwidth)
        print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size/MiB:.2f} MiB")
        evaluate(model, test_loader)
        quantizers[bitwidth] = quantizer
    accuracy_drop_threshold = 0.5
    quantizers_before_finetune = copy.deepcopy(quantizers)
    quantizers_after_finetune = quantizers
    for bitwidth in [8, 4, 2]:
        model.load_state_dict(torch.load("path/to/model", map_location="cpu")["model"])
        quantizer = quantizers[bitwidth]
        print(f'k-means quantizing model into {bitwidth} bits')
        quantizer.apply(model, update_centroids=False)
        quantized_model_size = get_model_size(model, bitwidth)
        print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size/MiB:.2f} MiB")
        quantized_model_accuracy = evaluate(model, test_loader)
        print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}% before quantization-aware training ")
        accuracy_drop = fp32_model_accuracy - quantized_model_accuracy
        if accuracy_drop > accuracy_drop_threshold:
            print(f"        Quantization-aware training due to accuracy drop={accuracy_drop:.2f}% is larger than threshold={accuracy_drop_threshold:.2f}%")
            num_finetune_epochs = 5
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
            criterion = nn.CrossEntropyLoss()
            best_accuracy = 0
            epoch = num_finetune_epochs
            while accuracy_drop > accuracy_drop_threshold and epoch > 0:
                train(model, train_loader, criterion, optimizer, scheduler,
                    callbacks=[lambda: quantizer.apply(model, update_centroids=True)])
                model_accuracy = evaluate(model, test_loader)
                is_best = model_accuracy > best_accuracy
                best_accuracy = max(model_accuracy, best_accuracy)
                print(f'        Epoch {num_finetune_epochs-epoch} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')
                accuracy_drop = fp32_model_accuracy - best_accuracy
                epoch -= 1
        else:
            print(f"        No need for quantization-aware training since accuracy drop={accuracy_drop:.2f}% is smaller than threshold={accuracy_drop_threshold:.2f}%")


if __name__ == "__main__":
    main()
    
