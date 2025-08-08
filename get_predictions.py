import torch
import pandas as pd
from tqdm import tqdm

from src.datasets.asv import TestASVspoof2019Dataset
from src.model.lcnn_model import LCNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = TestASVspoof2019Dataset()
protocol = open("ASVspoof2019.LA.cm.eval.trl.txt", "r")

model = LCNNModel(in_channels=1, num_classes=2, dropout_prob=0.75)
checkpoint = torch.load("saved/best.pth", map_location=device)
model.load_state_dict(
    checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
)
model.to(device)
model.eval()

labels = []
res_classes = []

for obj, name in tqdm(zip(data, protocol)):
    data_object = obj["data_object"].to(device, non_blocking=True)
    print(obj)
    break
    with torch.no_grad():
        output = model(data_object)
        pred = torch.argmax(output["logits"], dim=1).item()
        res_class = "bonafide" if pred else "spoof"
        labels.append(name.strip())
        res_classes.append(res_class)

df = pd.DataFrame(data={"label": labels, "prediction": res_classes})
df.to_csv("zzz.csv", index=False)
