import json
import math
import os

import numpy as np
from torch.utils.data import Dataset
import yaml

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.emotion_map = preprocess_config["emotions"]  # Load emotion mapping

        self.basename, self.speaker, self.text, self.phonemes, self.emotion, self.duration = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.text[idx]
        phone = np.array(text_to_sequence(self.phonemes[idx], self.cleaners))
        emotion = self.emotion[idx]
        duration = np.array(self.duration[idx])

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path) if os.path.exists(pitch_path) else np.zeros_like(mel.shape[0])
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path) if os.path.exists(energy_path) else np.zeros_like(mel.shape[0])

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "emotion": emotion,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            phonemes = []
            emotions = []
            durations = []
            for line in f.readlines():
                n, t, p, e, s, d = line.strip("\n").split("|")
                # Clean basename: use only filename without path or extension
                clean_basename = os.path.splitext(os.path.basename(n))[0]
                name.append(clean_basename)
                speaker.append(s)
                text.append(t)
                phonemes.append(p)
                emotions.append(self.emotion_map[e])
                durations.append([int(x) for x in d.split()])
            return name, speaker, text, phonemes, emotions, durations

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        speakers = np.array([data[idx]["speaker"] for idx in idxs])
        texts = [data[idx]["text"] for idx in idxs]  # List of phoneme sequences
        text_lens = np.array([len(data[idx]["text"]) for idx in idxs])
        max_text_len = max(text_lens)
        mels = [data[idx]["mel"] for idx in idxs]
        mel_lens = np.array([data[idx]["mel"].shape[0] for idx in idxs])
        max_mel_len = max(mel_lens)
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        emotions = np.array([data[idx]["emotion"] for idx in idxs])  # Scalars
        
        # Validate durations and texts
        for i, (dur, txt) in enumerate(zip(durations, texts)):
            if len(dur) != len(txt):
                print(f"Warning: Duration length {len(dur)} != text length {len(txt)} for id {ids[i]}")
                # Fallback: set durations to ones if text length is incorrect
                durations[i] = np.ones(len(txt))
        
        # Pad mel-spectrograms to max_mel_len
        mels_padded = []
        for mel in mels:
            pad_amount = max_mel_len - mel.shape[0]
            if pad_amount > 0:
                mel = np.pad(mel, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
            mels_padded.append(mel)
        mels = np.array(mels_padded)
        
        # Pad pitches to max_mel_len
        pitches_padded = []
        for pitch in pitches:
            pad_amount = max_mel_len - len(pitch)
            if pad_amount > 0:
                pitch = np.pad(pitch, (0, pad_amount), mode='constant', constant_values=0)
            pitches_padded.append(pitch)
        pitches = np.array(pitches_padded)
        
        # Pad energies to max_mel_len
        energies_padded = []
        for energy in energies:
            pad_amount = max_mel_len - len(energy)
            if pad_amount > 0:
                energy = np.pad(energy, (0, pad_amount), mode='constant', constant_values=0)
            energies_padded.append(energy)
        energies = np.array(energies_padded)
        
        durations = np.array(durations)
        texts = np.array(texts)
        
        # Pad sequences
        texts = pad_1D(texts, PAD=0)
        durations = pad_1D(durations, PAD=0)
        # No padding for emotions (scalars)
        
        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max_text_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            emotions,
        )
    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.emotion_map = preprocess_config["emotions"]

        self.basename, self.speaker, self.text, self.phonemes, self.emotion, self.duration = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.text[idx]
        phone = np.array(text_to_sequence(self.phonemes[idx], self.cleaners))
        emotion = self.emotion[idx]
        duration = np.array(self.duration[idx])

        return (basename, speaker_id, phone, raw_text, emotion, duration)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            phonemes = []
            emotions = []
            durations = []
            for line in f.readlines():
                n, t, p, e, s, d = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                phonemes.append(p)
                emotions.append(self.emotion_map[e])
                durations.append([int(x) for x in d.split()])
            return name, speaker, text, phonemes, emotions, durations

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        emotions = np.array([d[4] for d in data])
        durations = [d[5] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)
        durations = pad_1D(durations)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), emotions, durations


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/RAVDESS/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/RAVDESS/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train/train_metadata_with_durations.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "valid/valid_metadata_with_durations.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )