"""Frame-level sound event detectors (0.1 s resolution, 100 steps per 10 s scene).

Two interchangeable backends producing logits of shape (B, 100, n_classes):
  * CRNN      - small log-mel CNN + BiGRU trained from scratch (~0.35 M parameters; edge-sized).
  * ASTHead   - light BiGRU head on top of FROZEN AudioSet-pretrained AST patch tokens (features cached).
"""
import numpy as np
import torch
import torch.nn as nn

from .classes import N_CLASSES, SR, T_STEPS


class LogMel(nn.Module):
    def __init__(self, sr=SR, n_fft=1024, hop=160, n_mels=64, fmin=50, fmax=8000):
        super().__init__()
        import librosa
        mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer("mel", torch.tensor(mel, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(n_fft))
        self.n_fft, self.hop = n_fft, hop

    def forward(self, wav):                                   # (B, n) float in [-1, 1]
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window, center=True,
                          return_complex=True).abs() ** 2       # (B, F, T)
        return torch.log(torch.matmul(self.mel, spec) + 1e-6)   # (B, n_mels, T)


class CRNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES, n_mels=64, ch=(32, 64, 128), gru=64, dropout=0.3):
        super().__init__()
        self.front = LogMel(n_mels=n_mels)
        self.bn0 = nn.BatchNorm2d(1)

        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(pool))

        self.b1 = block(1, ch[0], (2, 2))       # (F/2, 500)
        self.b2 = block(ch[0], ch[1], (2, 5))   # (F/4, 100)
        self.b3 = block(ch[1], ch[2], (2, 1))   # (F/8, 100)
        self.proj = nn.Sequential(nn.Linear(ch[2] * (n_mels // 8), 128), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(128, gru, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * gru, n_classes)

    @staticmethod
    def spec_augment(x, fmax=8, tmax=60):
        b, _, f, t = x.shape
        x = x.clone()
        for i in range(b):
            fw = int(np.random.randint(0, fmax + 1)); f0 = int(np.random.randint(0, f - fw + 1))
            tw = int(np.random.randint(0, tmax + 1)); t0 = int(np.random.randint(0, t - tw + 1))
            x[i, :, f0:f0 + fw, :] = 0
            x[i, :, :, t0:t0 + tw] = 0
        return x

    def forward(self, wav):
        x = self.front(wav)[..., :T_STEPS * 10].unsqueeze(1)    # (B,1,F,1000)
        x = self.bn0(x)
        if self.training:
            x = self.spec_augment(x)
        x = self.b3(self.b2(self.b1(x)))                          # (B,C,F/8,100)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.gru(self.proj(x))
        return self.out(x)


class ASTHead(nn.Module):
    def __init__(self, in_dim=768, n_classes=N_CLASSES, hid=256, gru=128, dropout=0.3):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hid, gru, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * gru, n_classes)

    def forward(self, feats):                                  # (B, 100, in_dim)
        x, _ = self.gru(self.pre(feats))
        return self.out(x)


def build_model(backend, in_dim=768):
    return CRNN() if backend == "crnn" else ASTHead(in_dim=in_dim)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- frozen AST features
def load_ast(model_name="MIT/ast-finetuned-audioset-10-10-0.4593", random_init=False):
    """random_init=True builds a tiny random AST offline (used only by the unit test)."""
    from transformers import ASTConfig, ASTFeatureExtractor, ASTModel
    if random_init:
        cfg = ASTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
        return ASTFeatureExtractor(), ASTModel(cfg).eval()
    return ASTFeatureExtractor.from_pretrained(model_name), ASTModel.from_pretrained(model_name).eval()


@torch.no_grad()
def extract_ast_features(waves_int16, model_name, device="cpu", batch=16, random_init=False, log=print):
    """(N, 160000) int16 -> (N, 100, D) float16 time-resolved AST features.

    AST turns a 10.24 s log-mel patchified with stride 10 into a (12 freq x 101 time) grid of tokens.
    We average the token embeddings over frequency and keep the first 100 time steps (~0.1 s each).
    """
    fe, model = load_ast(model_name, random_init)
    model = model.to(device)
    cfg = model.config
    f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t_dim = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
    out = []
    for i in range(0, len(waves_int16), batch):
        chunk = [w.astype(np.float32) / 32768.0 for w in waves_int16[i:i + batch]]
        inp = fe(chunk, sampling_rate=SR, return_tensors="pt")["input_values"].to(device)
        use_amp = device != "cpu"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            h = model(inp).last_hidden_state
        tok = h[:, 2:, :].float()
        assert tok.shape[1] == f_dim * t_dim, (tok.shape, f_dim, t_dim)
        tok = tok.reshape(tok.shape[0], f_dim, t_dim, -1).mean(1)[:, :T_STEPS, :]
        out.append(tok.cpu().numpy().astype(np.float16))
        if (i // batch) % 20 == 0:
            log(f"  AST features {min(i + batch, len(waves_int16))}/{len(waves_int16)}")
    return np.concatenate(out)
