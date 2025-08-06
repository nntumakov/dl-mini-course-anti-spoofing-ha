import torch
from torch import nn


class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        num_channels = x.shape[1]

        if num_channels % 2 != 0:
            raise ValueError("num_channels % 2 != 0")

        split1, split2 = torch.split(x, num_channels // 2, dim=1)

        return torch.max(split1, split2)


class LCNNModel(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_prob):
        super().__init__()

        # ----

        self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.mfm_2 = MFM()

        # ----

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----

        self.conv_4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.mfm_5 = MFM()
        self.bn_6 = nn.BatchNorm2d(32)
        self.conv_7 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm_8 = MFM()

        # ----

        self.pool_9 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_10 = nn.BatchNorm2d(48)

        # ----

        self.conv_11 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
        self.mfm_12 = MFM()
        self.bn_13 = nn.BatchNorm2d(48)
        self.conv_14 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm_15 = MFM()

        # ----

        self.pool_16 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----

        self.conv_17 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.mfm_18 = MFM()
        self.bn_19 = nn.BatchNorm2d(64)
        self.conv_20 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm_21 = MFM()
        self.bn_22 = nn.BatchNorm2d(32)
        self.conv_23 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.mfm_24 = MFM()
        self.bn_25 = nn.BatchNorm2d(32)
        self.conv_26 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm_27 = MFM()

        # ----

        self.pool_28 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----

        self.fc_29 = nn.Linear(63936, 160)
        self.mfm_30 = MFM()
        self.bn_31 = nn.BatchNorm1d(80)

        # ----

        self.dropout = nn.Dropout(p=dropout_prob)

        # ----

        self.fc_32 = nn.Linear(80, num_classes)

    def forward(self, data_object, **batch):
        x = data_object

        x = self.conv_1(x)
        x = self.mfm_2(x)
        x = self.pool_3(x)
        x = self.conv_4(x)
        x = self.mfm_5(x)
        x = self.bn_6(x)
        x = self.conv_7(x)
        x = self.mfm_8(x)
        x = self.pool_9(x)
        x = self.bn_10(x)
        x = self.conv_11(x)
        x = self.mfm_12(x)
        x = self.bn_13(x)
        x = self.conv_14(x)
        x = self.mfm_15(x)
        x = self.pool_16(x)
        x = self.conv_17(x)
        x = self.mfm_18(x)
        x = self.bn_19(x)
        x = self.conv_20(x)
        x = self.mfm_21(x)
        x = self.bn_22(x)
        x = self.conv_23(x)
        x = self.mfm_24(x)
        x = self.bn_25(x)
        x = self.conv_26(x)
        x = self.mfm_27(x)
        x = self.pool_28(x)
        x = x.view(x.size(0), -1)
        x = self.fc_29(x)
        x = self.mfm_30(x)
        x = self.bn_31(x)
        x = self.dropout(x)

        logits = self.fc_32(x)

        return {"logits": logits}
