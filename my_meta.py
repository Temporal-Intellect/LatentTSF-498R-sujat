"""
Meta-correction of a trained LatentTSF forecaster in the latent residual space (498R-kdd style).

    Z_tilde_Y = Z_hat_Y - alpha * f_phi(R[t - L_m : t]),     Y_hat = D(Z_tilde_Y)

R = Z_hat_Y - Z_Y is the residual series built by my_residual.py. f_phi is a meta-corrector trained on it, by
default with the same backbone and hyperparameters as the base forecaster and a Huber loss. alpha is chosen on
the validation split. At a forecast origin t the meta-corrector reads only residual rows before t, which are
known by then (earlier base predictions minus truths that have already been observed).

The splits follow the base model: f_phi trains on the residual rows of the base train period, early-stops on the
validation period (also used to choose alpha) and is tested on the test period, where its windows coincide
one-to-one with the base model's test windows.

Run it with exactly the flags used for my_train.py and my_residual.py (see run_meta.sh), plus these optional ones:
  --residual_dir DIR       where my_residual.py wrote <DIR>/<model_id>/                (default: ./residuals/)
  --residual_name NAME     residual CSV name, <NAME>.csv                    (default: <model_id>_residual)
  --tsf_checkpoint PATH    base TSF checkpoint      (default: <checkpoints>/<setting>/checkpoint.pth)
  --itr_index K            base my_train.py iteration (the _K suffix of the setting)            (default: 0)
  --meta_args K=V ...      override the mirrored base config of f_phi, e.g. seq_len=96 model=iTransformer
  --meta_loss huber|mse    training loss of f_phi                                            (default: huber)
  --huber_delta D          Huber threshold                                                     (default: 1.0)
  --alpha_grid A,B,...     candidate alphas; 0 and 0.05 are always added          (default: 0,0.05,...,1)
  --meta_train 0|1         0: reuse the saved f_phi checkpoint and only evaluate                 (default: 1)
  --save_final_npy 0|1     save decoded test arrays at the chosen alpha                          (default: 1)

Outputs in <residual_dir>/<model_id>/meta_<loss>[_<overrides>]/:
  checkpoint.pth                 best f_phi (early stopping on the validation loss)
  alpha_sweep.csv                alpha vs. validation / test MSE, MAE (decoded, scaled) and latent MSE
  meta_info.json                 configs, training history, chosen alpha, before / after table, checks
  test_base_pred_<P>.npy, test_final_pred_<P>.npy, test_true_<P>.npy
                                 decoded base / corrected forecasts and truths, (windows, P, C), scaled space
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, Dataset

from my_utils import args_train, model_dict, set_seed, acquire_device, get_data, str2bool
from my_train import encoder_encode, encoder_decode, build_dec_inp_latent, build_dec_inp_original
from my_residual import SPLITS, check_args, training_setting, load_models
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.timefeatures import time_features

PAPER_ALPHA = 0.05
DEFAULT_ALPHAS = [round(0.05 * i, 2) for i in range(21)]
# Fixed by the alignment with the base model and the residual series
FIXED_META_KEYS = ('pred_len', 'enc_in', 'dec_in', 'c_out', 'features', 'data', 'root_path', 'data_path', 'target')


def parse_meta_args():
    """Read the meta-only flags; the rest of the command line goes to my_utils.args_train()."""
    if '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)
    # allow_abbrev=False: otherwise training flags such as --itr would be taken as one of these
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--residual_dir', type=str, default='./residuals/')
    parser.add_argument('--residual_name', type=str, default=None)
    parser.add_argument('--tsf_checkpoint', type=str, default=None)
    parser.add_argument('--itr_index', type=int, default=0)
    parser.add_argument('--meta_args', nargs='*', default=[])
    parser.add_argument('--meta_loss', type=str, default='huber', choices=['huber', 'mse'])
    parser.add_argument('--huber_delta', type=float, default=1.0)
    parser.add_argument('--alpha_grid', type=str, default=None)
    parser.add_argument('--meta_train', type=int, default=1)
    parser.add_argument('--save_final_npy', type=int, default=1)
    meta, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return meta


def parse_alphas(alpha_grid):
    alphas = DEFAULT_ALPHAS if alpha_grid is None else [float(a) for a in alpha_grid.split(',') if a.strip()]
    return sorted(set(round(a, 6) for a in alphas) | {0.0, PAPER_ALPHA})


def build_meta_args(args, overrides, latent_dim):
    """Config of f_phi: a copy of the base config with the --meta_args overrides applied."""
    meta_args = copy.copy(args)
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep or not hasattr(args, key):
            sys.exit(f'my_meta.py: --meta_args expects KEY=VALUE with a my_train.py argument name, got {item!r}')
        if key in FIXED_META_KEYS:
            sys.exit(f'my_meta.py: --meta_args cannot change {key}: it is fixed by the alignment with the base model')
        current = getattr(args, key)
        if isinstance(current, bool):
            value = str2bool(value)
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(meta_args, key, value)
    if meta_args.model not in model_dict:
        sys.exit(f'my_meta.py: unknown meta model {meta_args.model!r}; choose from {list(model_dict)}')
    # f_phi reads and predicts the D latent channels of the residual series
    meta_args.enc_in = meta_args.dec_in = meta_args.c_out = latent_dim
    return meta_args


def load_residual(args, meta, setting):
    """Residual series and its bookkeeping, checked against the current base run."""
    out_dir = os.path.join(meta.residual_dir, args.model_id)
    name = meta.residual_name or f'{args.model_id}_residual'
    paths = {k: os.path.join(out_dir, f'{name}{suffix}') for k, suffix in
             (('residual', '.csv'), ('ground_truth', '_latent_ground_truth.csv'), ('info', '_info.json'))}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        sys.exit(f'my_meta.py: residual series not found ({", ".join(missing)}).\n'
                 f'Run run_residual.sh first with the same settings.')
    with open(paths['info']) as f:
        info = json.load(f)
    expected = {'setting': setting, 'latent_dim': args.d_model, 'seq_len': args.seq_len,
                'pred_len': args.pred_len, 'residual_stride': 1}
    mismatched = {k: (info.get(k), v) for k, v in expected.items() if info.get(k) != v}
    if mismatched:
        sys.exit(f'my_meta.py: {paths["residual"]} was built for another run {mismatched}.\n'
                 f'Rerun run_residual.sh with the same settings.')

    residual = pd.read_csv(paths['residual'])
    z_cols = info['columns'][1:]
    if list(residual.columns) != ['date'] + z_cols:
        sys.exit(f'my_meta.py: unexpected columns in {paths["residual"]}')
    ground_truth = pd.read_csv(paths['ground_truth'], usecols=z_cols)[z_cols]
    splits = {csv_split: tuple(info['splits'][csv_split]['residual_rows']) for _, csv_split in SPLITS}
    return {
        'out_dir': out_dir,
        'paths': paths,
        'info': info,
        'values': residual[z_cols].values.astype(np.float32),
        'ground_truth': ground_truth.values.astype(np.float32),
        'dates': pd.to_datetime(residual['date']),
        'splits': splits,
    }


def time_marks(dates, meta_args):
    """Time features of the residual rows, built the way data_provider's loaders build them."""
    if meta_args.embed == 'timeF':
        return time_features(pd.DatetimeIndex(dates.values), freq=meta_args.freq).transpose(1, 0).astype(np.float32)
    columns = [dates.dt.month, dates.dt.day, dates.dt.weekday, dates.dt.hour]
    if meta_args.data in ('ETTm1', 'ETTm2'):
        columns.append(dates.dt.minute // 15)
    return np.stack([c.values for c in columns], axis=1).astype(np.float32)


class ResidualWindows(Dataset):
    """Windows of the residual series: input rows [q - seq_len, q), target rows [q - label_len, q + pred_len).

    Returns (x, y, x_mark, y_mark) like data_provider's datasets, so the usual forward pass applies.
    """

    def __init__(self, values, marks, origins, seq_len, label_len, pred_len):
        self.values, self.marks = values, marks
        self.origins = np.asarray(origins, dtype=np.int64)
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        if len(self.origins) and (self.origins.min() < seq_len or self.origins.max() + pred_len > len(values)):
            raise ValueError('residual windows fall outside the residual series')

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        q = int(self.origins[i])
        start, label_start, end = q - self.seq_len, q - self.label_len, q + self.pred_len
        return self.values[start:q], self.values[label_start:end], self.marks[start:q], self.marks[label_start:end]


def forward_meta(model, meta_args, batch_x, batch_y, batch_x_mark, batch_y_mark, device):
    """f_phi on a batch of residual windows (scaled), as in my_train.py's original mode."""
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    dec_inp = build_dec_inp_original(batch_y, meta_args.label_len, meta_args.pred_len, device)
    return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:, -meta_args.pred_len:, :]


def meta_loss_on(model, meta_args, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            outputs = forward_meta(model, meta_args, batch_x, batch_y, batch_x_mark, batch_y_mark, device)
            losses.append(criterion(outputs, batch_y[:, -meta_args.pred_len:, :].float().to(device)).item())
    model.train()
    return float(np.average(losses))


def train_meta(model, meta_args, train_set, val_set, criterion, ckpt_dir, device):
    """Same loop as my_train.py's original mode: Adam, gradient accumulation, LR schedule, early stopping."""
    train_loader = DataLoader(train_set, batch_size=meta_args.batch_size, shuffle=True,
                              num_workers=meta_args.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=meta_args.batch_size, shuffle=False,
                            num_workers=meta_args.num_workers, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=meta_args.learning_rate)
    early_stopping = EarlyStopping(patience=meta_args.patience, verbose=True)
    accum_steps = max(1, getattr(meta_args, 'accum_steps', 1))
    train_steps = len(train_loader)
    history = []

    for epoch in range(meta_args.train_epochs):
        model.train()
        train_losses = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if i % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            outputs = forward_meta(model, meta_args, batch_x, batch_y, batch_x_mark, batch_y_mark, device)
            loss = criterion(outputs, batch_y[:, -meta_args.pred_len:, :].float().to(device))
            train_losses.append(loss.item())
            window_size = min(accum_steps, train_steps - (i // accum_steps) * accum_steps)
            (loss / window_size).backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
        if train_steps % accum_steps != 0:
            optimizer.step()

        val_loss = meta_loss_on(model, meta_args, val_loader, criterion, device)
        history.append({'epoch': epoch + 1, 'train_loss': float(np.average(train_losses)), 'val_loss': val_loss,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        print(f'Meta epoch {epoch + 1}: train loss {history[-1]["train_loss"]:.7f}, val loss {val_loss:.7f}')
        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        if meta_args.use_lradj:
            adjust_learning_rate(optimizer, epoch + 1, meta_args)
    return history


def evaluate_correction(args, meta_args, flag, rows, base_models, meta_model, residual, scaled, marks, scaler_t,
                        alphas, device, save_alpha=None, npy_paths=None):
    """Decoded MSE / MAE of D(Z_hat_Y - alpha * R_hat) for every alpha, over every window of one base split.

    Base window i (stride 1) and the residual window with origin rows[0] + i forecast the same rows, so the two
    loaders are walked in lockstep.
    """
    encoder, model, latent_norm = base_models
    pred_len, d_latent = args.pred_len, args.d_model
    base_set, _ = get_data(args, flag=flag)  # args.step == 1: every window, in order
    n_windows = len(base_set)
    if rows[0] + n_windows - 1 + pred_len != rows[1]:
        raise RuntimeError(f'{flag}: {n_windows} base windows do not cover residual rows [{rows[0]}, {rows[1]})')
    origins = rows[0] + np.arange(n_windows)
    meta_set = ResidualWindows(scaled, marks, origins, meta_args.seq_len, meta_args.label_len, pred_len)
    base_loader = DataLoader(base_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)
    meta_loader = DataLoader(meta_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    arrays = None
    if npy_paths is not None:
        shape = (n_windows, pred_len, args.c_out)
        arrays = [open_memmap(p, mode='w+', dtype=np.float32, shape=shape) for p in npy_paths]

    point_wise = args.ae_type.upper() == 'MLP'
    sq_err = np.zeros(len(alphas))
    abs_err = np.zeros(len(alphas))
    latent_sq_err = np.zeros(len(alphas))
    n_obs = n_latent = 0
    alignment_gap = 0.0
    w = 0
    meta_model.eval()
    with torch.no_grad():
        for (batch_x, batch_y, batch_x_mark, batch_y_mark), meta_batch in zip(base_loader, meta_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            y_target = batch_y[:, -pred_len:, :]

            # Base forecast, exactly as my_train.py / my_residual.py
            z_x = encoder_encode(encoder, batch_x, 'AE', latent_norm)
            z_y = encoder_encode(encoder, y_target, 'AE', None)
            dec_inp = build_dec_inp_latent(z_x, args.label_len, pred_len, d_latent, device)
            z_pred = model(z_x, batch_x_mark, dec_inp, batch_y_mark)[:, -pred_len:, :]

            # Residual forecast from the residual rows before each origin, back in latent units
            r_hat = forward_meta(meta_model, meta_args, *meta_batch, device) * scaler_t[1] + scaler_t[0]

            b = z_pred.shape[0]
            if point_wise:  # E(Y) of this window must equal the latent ground truth at rows [q, q + P)
                rows = origins[w:w + b, None] + np.arange(pred_len)
                alignment_gap = max(alignment_gap,
                                    float(np.abs(z_y.cpu().numpy() - residual['ground_truth'][rows]).max()))

            for k, alpha in enumerate(alphas):
                z_corr = z_pred - alpha * r_hat
                y_hat = encoder_decode(encoder, z_corr, 'AE')
                err = (y_hat - y_target).double()
                sq_err[k] += err.pow(2).sum().item()
                abs_err[k] += err.abs().sum().item()
                latent_sq_err[k] += (z_corr - z_y).double().pow(2).sum().item()
                if arrays is not None and alpha == save_alpha:
                    arrays[1][w:w + b] = y_hat.cpu().numpy()
                if arrays is not None and alpha == 0.0:
                    arrays[0][w:w + b] = y_hat.cpu().numpy()
            if arrays is not None:
                arrays[2][w:w + b] = y_target.cpu().numpy()
            n_obs += y_target.numel()
            n_latent += z_y.numel()
            w += b

    if w != n_windows:
        raise RuntimeError(f'{flag}: {w}/{n_windows} windows evaluated')
    if point_wise and alignment_gap > 1e-4:
        raise RuntimeError(f'{flag}: base windows and residual rows are misaligned (max |E(Y) - Z_Y| = {alignment_gap})')
    if arrays is not None:
        for a in arrays:
            a.flush()
        del arrays
    return {
        'windows': n_windows,
        'mse': sq_err / n_obs,
        'mae': abs_err / n_obs,
        'latent_mse': latent_sq_err / n_latent,
        'alignment_max_abs_diff': alignment_gap if point_wise else None,
    }


def main():
    meta = parse_meta_args()
    args = args_train()
    check_args(args)

    set_seed(args.seed)
    if args.use_gpu and args.gpu_type == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available: using the CPU')
        args.use_gpu = False
    device = acquire_device(args)

    # Base run (same checkpoint lookup as my_residual.py), then every evaluation window with stride 1
    setting = training_setting(args, meta.itr_index)
    tsf_checkpoint = meta.tsf_checkpoint or os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(tsf_checkpoint):
        sys.exit(f'my_meta.py: base TSF checkpoint not found:\n  {tsf_checkpoint}\n'
                 f'Train first with the same flags (run_train.sh), or pass --tsf_checkpoint.')
    latent_dim = args.d_model
    meta_args = build_meta_args(args, meta.meta_args, latent_dim)  # mirrors the base config, training --step
    args.step = 1

    residual = load_residual(args, meta, setting)
    splits = residual['splits']
    alphas = parse_alphas(meta.alpha_grid)
    overrides = ''.join(f'_{k}{v}' for k, v in sorted(item.split('=', 1) for item in meta.meta_args))
    meta_dir = os.path.join(residual['out_dir'], f'meta_{meta.meta_loss}{overrides}'.replace('/', '-'))
    ckpt_path = os.path.join(meta_dir, 'checkpoint.pth')
    if not meta.meta_train and not os.path.exists(ckpt_path):
        sys.exit(f'my_meta.py: --meta_train 0 but no meta checkpoint at {ckpt_path}')
    os.makedirs(meta_dir, exist_ok=True)

    print(f'Base setting  : {setting}')
    print(f'Residual CSV  : {residual["paths"]["residual"]}  rows={len(residual["values"])}, D={latent_dim}')
    print(f'Meta-corrector: {meta_args.model}, seq_len={meta_args.seq_len}, d_model={meta_args.d_model}, '
          f'loss={meta.meta_loss} -> {meta_dir}')
    base_models = load_models(args, tsf_checkpoint, device)

    # Residual series, scaled with statistics of the meta-train rows only
    train_rows, val_rows, test_rows = splits['train'], splits['vali'], splits['test']
    scaler = StandardScaler().fit(residual['values'][train_rows[0]:train_rows[1]])
    scaled = scaler.transform(residual['values']).astype(np.float32)
    scaler_t = (torch.tensor(scaler.mean_, dtype=torch.float32, device=device),
                torch.tensor(scaler.scale_, dtype=torch.float32, device=device))
    marks = time_marks(residual['dates'], meta_args)

    # Origins in residual-row coordinates: targets stay inside one base split, inputs only look back
    seq_len, label_len, pred_len = meta_args.seq_len, meta_args.label_len, args.pred_len
    train_origins = np.arange(max(seq_len, train_rows[0]), train_rows[1] - pred_len + 1, max(1, meta_args.step))
    val_origins = np.arange(val_rows[0], val_rows[1] - pred_len + 1, max(1, meta_args.step))
    if len(train_origins) == 0 or val_rows[0] < seq_len:
        sys.exit(f'my_meta.py: the train period ({train_rows[1] - train_rows[0]} residual rows) is too short '
                 f'for seq_len={seq_len} + pred_len={pred_len}')
    train_set = ResidualWindows(scaled, marks, train_origins, seq_len, label_len, pred_len)
    val_set = ResidualWindows(scaled, marks, val_origins, seq_len, label_len, pred_len)

    set_seed(args.seed)
    meta_model = model_dict[meta_args.model].Model(meta_args).float().to(device)
    criterion = nn.HuberLoss(delta=meta.huber_delta) if meta.meta_loss == 'huber' else nn.MSELoss()
    history = None
    if meta.meta_train:
        print(f'Training f_phi on {len(train_set)} windows (validation: {len(val_set)} windows)')
        history = train_meta(meta_model, meta_args, train_set, val_set, criterion, meta_dir, device)
    meta_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    meta_model.eval()

    # Choose alpha on the validation split, then score the test split.
    # alpha = 0 is the base model: it must reproduce the decoded metrics my_residual.py recorded.
    k0 = alphas.index(0.0)
    recorded = residual['info']['checks']['decoded_metrics']
    base_gaps = {}

    def check_base(split, result):
        gap = max(abs(result['mse'][k0] - recorded[split]['mse']), abs(result['mae'][k0] - recorded[split]['mae']))
        base_gaps[split] = float(gap)
        if gap > 1e-5:
            raise RuntimeError(f'alpha=0 does not reproduce the base model on {split} (diff {gap:.2e}): the residual '
                               f'series is stale; rerun run_residual.sh after retraining the base model')

    val = evaluate_correction(args, meta_args, 'val', val_rows, base_models, meta_model, residual, scaled,
                              marks, scaler_t, alphas, device)
    check_base('val', val)
    best = int(np.argmin(val['mse']))
    alpha_star = alphas[best]
    npy_paths = None
    if meta.save_final_npy:
        npy_paths = [os.path.join(meta_dir, f'test_{kind}_{pred_len}.npy') for kind in ('base_pred', 'final_pred', 'true')]
    test = evaluate_correction(args, meta_args, 'test', test_rows, base_models, meta_model, residual, scaled,
                               marks, scaler_t, alphas, device, save_alpha=alpha_star, npy_paths=npy_paths)
    check_base('test', test)

    sweep = pd.DataFrame({'alpha': alphas,
                          'val_mse': val['mse'], 'val_mae': val['mae'], 'val_latent_mse': val['latent_mse'],
                          'test_mse': test['mse'], 'test_mae': test['mae'], 'test_latent_mse': test['latent_mse']})
    sweep_path = os.path.join(meta_dir, 'alpha_sweep.csv')
    sweep.to_csv(sweep_path, index=False)

    # Before / after table
    rows = [('base (alpha=0)', 0.0), (f'paper (alpha={PAPER_ALPHA})', PAPER_ALPHA), ('chosen on val', alpha_star)]
    table = []
    print(f'\n{"":24s}{"alpha":>7s}{"val MSE":>11s}{"val MAE":>11s}{"test MSE":>11s}{"test MAE":>11s}')
    for label, alpha in rows:
        k = alphas.index(alpha)
        table.append({'row': label, 'alpha': alpha, 'val_mse': float(val['mse'][k]), 'val_mae': float(val['mae'][k]),
                      'test_mse': float(test['mse'][k]), 'test_mae': float(test['mae'][k])})
        print(f'{label:24s}{alpha:7.2f}{val["mse"][k]:11.6f}{val["mae"][k]:11.6f}'
              f'{test["mse"][k]:11.6f}{test["mae"][k]:11.6f}')

    print(f'\nalpha=0 reproduces the base model (vs. my_residual.py): max diff val {base_gaps["val"]:.2e}, '
          f'test {base_gaps["test"]:.2e}')
    print(f'Saved {sweep_path}')

    info = {
        'formula': 'Y_hat = D(Z_hat_Y - alpha * f_phi(R[t - L_m : t]))',
        'base_setting': setting,
        'tsf_checkpoint': tsf_checkpoint,
        'autoencoder_path': args.autoencoder_path,
        'residual_csv': residual['paths']['residual'],
        'meta_checkpoint': ckpt_path,
        'meta_config': {
            'model': meta_args.model, 'seq_len': seq_len, 'label_len': label_len, 'pred_len': pred_len,
            'channels': latent_dim, 'd_model': meta_args.d_model, 'd_ff': meta_args.d_ff,
            'e_layers': meta_args.e_layers, 'n_heads': meta_args.n_heads, 'dropout': meta_args.dropout,
            'batch_size': meta_args.batch_size, 'learning_rate': meta_args.learning_rate,
            'lradj': meta_args.lradj, 'use_lradj': meta_args.use_lradj, 'train_epochs': meta_args.train_epochs,
            'patience': meta_args.patience, 'accum_steps': getattr(meta_args, 'accum_steps', 1),
            'step': meta_args.step, 'loss': meta.meta_loss, 'huber_delta': meta.huber_delta,
            'overrides': meta.meta_args,
        },
        'windows': {'train': len(train_set), 'val_early_stopping': len(val_set),
                    'val': val['windows'], 'test': test['windows']},
        'scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()},
        'history': history,
        'alpha_grid': alphas,
        'alpha_star': alpha_star,
        'table': table,
        'checks': {
            'alpha0_vs_base_max_abs_diff': base_gaps,
            'alignment_max_abs_diff': {'val': val['alignment_max_abs_diff'], 'test': test['alignment_max_abs_diff']},
        },
        'files': {'alpha_sweep': sweep_path, 'final_npy': npy_paths},
    }
    info_path = os.path.join(meta_dir, 'meta_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'Saved {info_path}')

    if args.gpu_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
