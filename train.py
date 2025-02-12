import os
import json
import subprocess
import pandas as pd
from tqdm import tqdm
import torchaudio
import whisper
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from args import get_config
from utils import save_ckpt, load_ckpt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from collections import namedtuple
from fast_pytorch_kmeans import KMeans
from torch.nn import parameter


Codebook = namedtuple('Codebook', ['centroids', 'labels'])


def add_intent(df, filename):
    intent = []
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i]
        action = row['action'] if row['action'] != "none" else ""
        object = row['object'] if row['object'] != "none" else ""
        location = row['location'] if row['location'] != "none" else ""
        intent.append(f"{action} {object}  {location}".strip())
    df["intent"] = intent
    df.to_csv(f"./data/fluent_speech_commands/data/{filename}", index=False)


def json_to_dict(file_location):
    with open(file_location, "r") as json_file:
        data_dict = json.load(json_file)
    return data_dict


class FSC:
    def __init__(
        self,
        loader_type
    ):
        self.data_root_dir = "./data/fluent_speech_commands"
        self.loader_type = loader_type
        self.meta_data_dir = os.path.join(self.data_root_dir, f"data/{self.loader_type}_data.csv")
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


class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = whisper.load_model("large").encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, dilation=128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = torch.mean(x, 1)
        return x




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


def quantization_phase(model, loader, callbacks, it, writer, epoch, max_epoch, num_epochs=3):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for inputs, targets in tqdm(loader, desc=f"Quantization: {epoch + 1}/{max_epoch}"):
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
        writer.add_scalar("Loss/CE Loss", loss.item(), it)
        it += 1
    return model, it


def main(config):
    output = subprocess.check_output("nvidia-smi", shell=True)
    output = output.decode("utf-8")
    print(output)
    print("\n\n\n")
    batch_size = 32
    dataset = FSC(loader_type=config.mode)
    loader = load_data(dataset=dataset, batch_size=batch_size, shuffle=config.shuffle)
    
    model = WhisperBaselineModel().to(device=config.device)
    if config.start_from != 0:
        load_ckpt(model=model, config=config)
    loss_fn = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.intent_classifier.parameters(), "lr": 1e-3},
        ]
    )
    max_epoch = 20
    bitwidth = 8
    quantizer = KMeansQuantizer(model, bitwidth)
    teacher = TeacherModel().to(device=config.device)
    distilattion = [1, 2, 3, 7, 8, 9, 10, 11, 12]
    quantization = [4, 5, 6, 13, 14, 15]
    alpha = 1e-6 
    if config.mode == "train":
        writer = SummaryWriter(log_dir=os.path.join(config.config_path, f"training_restarted_at_epoch_{config.start_from}"))
        it = 0
        for epoch in range(config.start_from, max_epoch):
            if (epoch+1) in distilattion:
                for feature, intent in tqdm(loader, desc=f"Distillation: {epoch+1} / {max_epoch}"):
                    optimizer.zero_grad()
                    feature = feature.to(config.device)
                    intent = intent.to(config.device)
                    with torch.no_grad():
                        teacher_out = teacher(feature)
                    student_out, intent_pred = model(feature)
                    l1 = l1_loss(teacher_out, student_out)
                    ce = loss_fn(intent_pred, intent)
                    # loss = loss_fn(intent_pred, intent)
                    loss = alpha*l1 + ce
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar("Loss/CE Loss", ce.item(), it)
                    writer.add_scalar("Loss/L1 Loss", l1.item(), it)
                    writer.add_scalar("Loss/Total Loss", loss.item(), it)
                    it += 1
            else:
                model, it = quantization_phase(
                    model, 
                    loader, 
                    writer=writer, 
                    it=it,
                    epoch=epoch,
                    max_epoch=max_epoch,
                    callbacks=[lambda: quantizer.apply(model, update_centroids=True)]
                )
            save_ckpt(
                epoch=epoch+1,
                model=model,
                config=config
            )
    else:
        model = load_ckpt(model=model, config=config)
        model = model.to(device=config.device)
        model.eval()
        preds = []
        gt = []
        for feature, intent in tqdm(loader, desc=f"Evaluating"):
            feature = feature.to(config.device)
            with torch.no_grad():
                _, intent_pred = model(feature)
                gt.append(intent)
                preds.append(intent_pred)
        preds = torch.cat(preds, dim=0)
        gt = torch.cat(gt, dim=0)
        preds = torch.argmax(preds, axis=1)
        gt = gt.to("cpu")
        preds = preds.to("cpu")
        accuracy = round(accuracy_score(y_true=gt, y_pred=preds), 4)
        precision = round(precision_score(y_true=gt, y_pred=preds, average="macro"), 4)
        recall = round(recall_score(y_true=gt, y_pred=preds, average="macro"), 4)
        f1 = round(f1_score(y_true=gt, y_pred=preds, average="macro"), 4)
        txt_file_name = f"evaluation_report{config.ckpt_name}.txt"
        with open(os.path.join(config.log_dir, txt_file_name), "a") as file:
            print(f"Accuracy: {accuracy}", file=file)
            print(f"Precision: {precision}", file=file)
            print(f"Recall: {recall}", file=file)
            print(f"F1-Score: {f1}", file=file)
            print("="*25, file=file)
            print("\n\n\n", file=file)
            print(
                classification_report(y_true=gt, y_pred=preds),
                file=file
            )
        file.close()
        

def teacher_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 80, 3000).to(device)
    teacher = TeacherModel().to(device)
    y = teacher(x)
    print(teacher)
    print(y.shape)


if __name__ == "__main__":
    config = get_config()
    main(config)
