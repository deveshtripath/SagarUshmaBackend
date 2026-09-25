"""Architecture rebuilt from the weight names/shapes stored in last_model.pt.

Layer NAMES and SHAPES are exact (load_state_dict checks every one).
Things a checkpoint cannot tell us, so they are best guesses -- compare with
your training notebook: strides (2 in each encoder conv), activations (ReLU),
modality order, how depth is added in the decoder, and the ConvGRU update rule.
"""
import torch
import torch.nn as nn

# Order matters: it is the channel order of the fusion gate (5 x 32 = 160).
MODALITIES = ["sst", "sss", "ssh", "current", "wind"]
CHANNELS = {"sst": 3, "sss": 1, "ssh": 1, "current": 3, "wind": 3}
OUTPUT_CHANNELS = ["channel_0", "channel_1"]  # rename once you know what they are


# ---- Fill these in with what your notebook does (currently pass-through) ----
def prepare_input(name, x):
    """x: numpy array (T, C, H, W) for one variable. Apply your normalization here."""
    return x


def prepare_depth(depth):
    """Depth in the units/scaling the model was trained with."""
    return depth


def finish_output(y):
    """y: numpy array (2, H, W). Undo target normalization here."""
    return y


# ---------------------------------- model ------------------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(160, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5, 1),
            nn.Softmax(dim=1),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        w = self.gate(torch.cat(feats, dim=1))
        fused = sum(w[:, i:i + 1] * f for i, f in enumerate(feats))
        return self.output(fused)


class ConvGRUCell(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.gates = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1)
        self.candidate = nn.Conv2d(2 * ch, ch, 3, padding=1)

    def forward(self, x, h):
        r, z = torch.sigmoid(self.gates(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        cand = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * cand


class TemporalModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = ConvGRUCell(128)

    def forward(self, seq):
        h = torch.zeros_like(seq[0])
        for x in seq:
            h = self.cell(x, h)
        return h


class DepthEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, d):
        return self.network(d)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_embedding = DepthEmbedding()
        self.depth_projection = nn.Linear(32, 32)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(32, 2, 1)

    def forward(self, h, depth):
        x = self.up2(self.up1(h))
        e = self.depth_projection(self.depth_embedding(depth.view(-1, 1)))
        x = x + e[:, :, None, None]
        return self.output(self.refine(x))


class OceanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modalities = nn.ModuleDict(
            {f"{m}_encoder": ModalityEncoder(CHANNELS[m]) for m in MODALITIES}
        )
        self.fusion = GatedFusion()
        self.temporal = TemporalModule()
        self.decoder = Decoder()

    def forward(self, inputs, depth):
        """inputs: {name: (B, T, C, H, W)}, depth: (B,) -> (B, 2, H, W)"""
        steps = inputs[MODALITIES[0]].shape[1]
        seq = []
        for t in range(steps):
            feats = [self.modalities[f"{m}_encoder"](inputs[m][:, t]) for m in MODALITIES]
            seq.append(self.fusion(feats))
        return self.decoder(self.temporal(seq), depth)


def build_model():
    return OceanModel()
