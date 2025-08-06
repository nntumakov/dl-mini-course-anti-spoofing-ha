from pathlib import Path
from typing import Callable, Dict

import torch
import torchaudio
import torch.nn.functional as F

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

# -- Constants --

WINDOW_SIZE: int = 1726  # выбрал, чтобы получить размерность 864 * ...
MAX_LEN: int = 600  # выбрал, чтобы получить размерность 864 * 600


class STFTTransform:
    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=torch.blackman_window(self.win_length),
        )

        magnitude = torch.abs(stft)

        log_magnitude = torch.log1p(magnitude)

        return log_magnitude.squeeze(0)


class PadAndTrimTransform:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        current_len = tensor.shape[-1]

        if current_len > self.max_len:
            res_tensor = tensor[..., : self.max_len]
        elif current_len < self.max_len:
            padding_needed = self.max_len - current_len
            res_tensor = F.pad(tensor, (0, padding_needed), "constant", 0)
        else:
            res_tensor = tensor

        return res_tensor.unsqueeze(0).unsqueeze(0)


class ASVspoof2019Dataset(BaseDataset):
    def __init__(self, part: str):
        data_path = ROOT_PATH / "data"

        index = self._create_index(data_path, part)

        window_size: int = WINDOW_SIZE

        transforms: Dict[str, Callable] = {
            "stftt": STFTTransform(
                n_fft=window_size,
                hop_length=int(window_size / 4),
                win_length=window_size,
            ),
            "data_object": PadAndTrimTransform(max_len=MAX_LEN),
        }

        super().__init__(index=index, instance_transforms=transforms)

    def _create_index(self, data_path: Path, part: str) -> list:
        protocol_filename = (
            f"ASVspoof2019.LA.cm.{part}.{'trn' if part == 'train' else 'trl'}.txt"
        )
        
        protocol_path = (
            data_path / "LA" / "LA" / "ASVspoof2019_LA_cm_protocols" / protocol_filename
        )

        flac_path = data_path / "LA" / f"ASVspoof2019_LA_{part}" / "flac"

        index = []

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                _, obj_id, _, _, label = parts

                file_path = flac_path / f"{obj_id}.flac"

                index.append(
                    {
                        "path": file_path,
                        "label": 1 if label == "bonafide" else 0,
                    }
                )

        return index

    def load_object(self, path: str) -> torch.Tensor:
        waveform, _ = torchaudio.load(path)
        return waveform


class TrainASVspoof2019Dataset(ASVspoof2019Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(part="train")


class DevASVspoof2019Dataset(ASVspoof2019Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(part="dev")


class TestASVspoof2019Dataset(ASVspoof2019Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(part="eval")
