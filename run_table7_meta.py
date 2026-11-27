#!/usr/bin/env python3
"""
Every experiment of the LatentTSF paper's Table 7 (appendix E) through the meta-correction pipeline:

    6 datasets (ETTm1, ETTm2, ETTh1, ETTh2, Traffic, Electricity) x 6 backbones (CMoS, DLinear, PatchTST,
    TimeBase, TimeXer, iTransformer) x 4 horizons (96, 192, 336, 720) = 144 cells

Each cell runs the three pipeline scripts, unchanged and with the same flags:
    run_train.sh     trains the LatentTSF forecaster (paper recipe and per-dataset settings)
    run_residual.sh  builds its latent residual series R = Z_hat_Y - Z_Y
    run_meta.sh      trains the meta-corrector f_phi (same backbone as the forecaster) and scores
                     Y_hat = D(Z_hat_Y - alpha * f_phi(R)) at alpha = 0, 0.05 and alpha* chosen on validation
The paper's own columns (Original, w/LatentTSF) are not rerun.

Storage (Kaggle: 19.5 GB). The sweep never writes the large per-window arrays (--save_window_npy 0 and
--save_final_npy 0, which would reach several GB per array on Traffic and Electricity). Once a cell's metrics are
collected it deletes the residual CSVs and the f_phi checkpoint. By default it keeps only the forecaster
checkpoints (about 1.2 GB for all 144 cells) and small files. One cell needs about 1.5 GB at its peak
(Electricity, Traffic). Before each cell, the disk used under --disk_root is checked against
--disk_limit_gb minus --disk_margin_gb.

Resuming. Each cell keeps its state in <out_dir>/cells/<cell>/status.json. Finished cells are skipped, an
unfinished cell restarts after its last completed stage, and failed cells are retried only with --retry_failed.
Cell folders are self-contained: to combine several sessions, copy their cell folders into one <out_dir>/cells/
and run --collect_only.

Outputs in <out_dir> (default ./sweeps/table7_meta/):
  results.csv              one row per cell: status, val/test MSE and MAE at alpha = 0, 0.05 and alpha*, alpha*, ...
  table7_meta.md           one table per backbone: LatentTSF (ours, alpha = 0), +Meta alpha = 0.05, +Meta alpha*
  table7_meta.tex          Table 7's layout with the two meta-corrected column groups per backbone
  cells/<cell>/            status.json, meta_info.json, alpha_sweep.csv, residual_info.json, {train,residual,meta}.log
  checkpoints/             forecaster checkpoints (--keep forecaster, the default)
  train_results.csv / .txt my_train.py's own test metrics (they use the training --step)
"""
import argparse
import csv
import glob
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))

# Paper Table 7: dataset row order, backbone column order and horizons
TABLE7_DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'traffic', 'electricity']
TABLE7_MODELS = ['CMoS', 'DLinear', 'PatchTST', 'TimeBase', 'TimeXer', 'iTransformer']
TABLE7_PRED_LENS = [96, 192, 336, 720]
RUN_ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'traffic']  # smaller datasets first
DISPLAY_NAME = {'traffic': 'Traffic', 'electricity': 'Electricity'}
PAPER_ALPHA = 0.05
GB = 1e9
# Set by the sweep itself, so not allowed in --extra / --meta_extra
MANAGED_FLAGS = ('--dataset', '--model', '--pred_len', '--model_id', '--checkpoints', '--residual_dir',
                 '--save_window_npy', '--save_final_npy', '--result_csv', '--result_txt', '--num_workers')
# Written by my_residual.py but not read by my_meta.py: deleted right after the residual step
UNUSED_RESIDUAL_FILES = ('train_predictions.csv', 'vali_predictions.csv', 'test_predictions.csv',
                         '*_merged_predictions.csv')
VARIANTS = (('base', 'alpha = 0'), ('a005', f'alpha = {PAPER_ALPHA}'), ('astar', 'alpha*'))
RESULT_COLUMNS = (
    ['dataset', 'model', 'pred_len', 'status', 'failed_stage', 'alpha_star']
    + [f'test_{m}_{v}' for v, _ in VARIANTS for m in ('mse', 'mae')]
    + [f'val_{m}_{v}' for v, _ in VARIANTS for m in ('mse', 'mae')]
    + ['train_test_mse', 'train_test_mae', 'train_step', 'train_epochs', 'meta_epochs', 'test_windows',
       'check_alpha0_val', 'check_alpha0_test', 'seconds_train', 'seconds_residual', 'seconds_meta',
       'extra_flags', 'meta_extra_flags', 'finished_at', 'forecaster_checkpoint', 'git_commit', 'error'])

EXAMPLES = """examples:
  python run_table7_meta.py --dry_run          # list the cells and the three commands of each
  python run_table7_meta.py --datasets ETTh1 --models DLinear --pred_lens 96 --extra="--train_epochs 1" \\
                            --out_dir sweeps/smoke                # quick check of the whole chain
  python run_table7_meta.py --max_hours 10     # one 12-hour Kaggle session; run it again to continue
  python run_table7_meta.py --datasets ETTh1 ETTh2 --models DLinear iTransformer
  python run_table7_meta.py --datasets traffic --models PatchTST --retry_failed \\
                            --extra="--batch_size 8 --accum_steps 16"   # rerun cells that ran out of GPU memory
  python run_table7_meta.py --collect_only     # rebuild results.csv and the tables

Pass --extra / --meta_extra with "=", e.g. --extra="--train_epochs 1", so the value is not read as a flag."""


# ----------------------------------------------------------------------------------------------- small helpers
def now():
    return datetime.now().isoformat(timespec='seconds')


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    """Write through a temporary file, so an interrupted write never leaves a broken file behind."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def remove(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.lexists(path):
        os.remove(path)


def disk_used(root):
    """Bytes in regular files under root (apparent size, the conservative measure)."""
    total = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
    return total


def default_disk_root():
    """/kaggle/working when the repository is inside it (that is what Kaggle's 19.5 GB limit counts)."""
    kaggle = os.path.realpath('/kaggle/working')
    repo = os.path.realpath(REPO)
    if os.path.isdir(kaggle) and (repo == kaggle or repo.startswith(kaggle + os.sep)):
        return kaggle
    return REPO


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def log_tail(path, lines=3):
    try:
        with open(path, errors='replace') as f:
            text = [line.rstrip() for line in f if line.strip()]
    except OSError:
        return ''
    return ' | '.join(text[-lines:])[-600:]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    number = to_float(value)
    return None if number is None else int(number)


def duration(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}h{minutes:02d}m' if hours else (f'{minutes}m{secs:02d}s' if minutes else f'{secs}s')


def pick(values, allowed, what):
    """Selection from a Table 7 list, case-insensitive, comma or space separated; canonical spelling, list order."""
    if not values:
        return list(allowed)
    lookup = {str(a).lower(): a for a in allowed}
    lookup.update({DISPLAY_NAME[a].lower(): a for a in allowed if a in DISPLAY_NAME})
    chosen = set()
    for value in values:
        for part in str(value).replace(',', ' ').split():
            if part.lower() not in lookup:
                sys.exit(f'run_table7_meta.py: {part!r} is not a Table 7 {what} '
                         f'({", ".join(str(a) for a in allowed)})')
            chosen.add(lookup[part.lower()])
    return [a for a in allowed if a in chosen]


def split_flags(text, option):
    flags = shlex.split(text or '')
    for token in flags:
        if token.split('=', 1)[0] in MANAGED_FLAGS:
            sys.exit(f'run_table7_meta.py: {option} cannot contain {token.split("=", 1)[0]}: the sweep sets it')
    return flags


# ----------------------------------------------------------------------------------------------- the sweep
class Sweep:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out_dir)
        self.cells_dir = os.path.join(self.out, 'cells')
        self.ckpt_root = os.path.join(self.out, 'checkpoints')
        self.work = os.path.abspath(args.work_dir) if args.work_dir else os.path.join(self.out, 'work')
        self.train_csv = os.path.join(self.out, 'train_results.csv')
        self.train_txt = os.path.join(self.out, 'train_results.txt')
        self.disk_root = os.path.abspath(args.disk_root) if args.disk_root else default_disk_root()
        self.extra = split_flags(args.extra, '--extra')
        self.meta_extra = split_flags(args.meta_extra, '--meta_extra')
        self.flags = {'extra': shlex.join(self.extra), 'meta_extra': shlex.join(self.meta_extra)}
        self.env = dict(os.environ)
        self.env.setdefault('WANDB_MODE', 'disabled')
        self.commit = git_commit()

    # ---- cells
    @staticmethod
    def cells(datasets, models, pred_lens):
        return [{'cell': f'{d}_{m}_pl{p}', 'dataset': d, 'model': m, 'pred_len': p,
                 'model_id': f'Latent_{m}_{d}_pl{p}'}
                for d in RUN_ORDER if d in datasets for m in models for p in pred_lens]

    def status_path(self, cell):
        return os.path.join(self.cells_dir, cell['cell'], 'status.json')

    def status(self, cell):
        return read_json(self.status_path(cell), {}) or {}

    def commands(self, cell):
        common = ['--dataset', cell['dataset'], '--model', cell['model'], '--pred_len', str(cell['pred_len']),
                  '--checkpoints', self.ckpt_root + os.sep, '--num_workers', str(self.args.num_workers)]
        common += self.extra
        work = self.work + os.sep
        return {
            'train': ['bash', 'run_train.sh', *common, '--result_csv', self.train_csv, '--result_txt', self.train_txt],
            'residual': ['bash', 'run_residual.sh', *common, '--save_window_npy', '0', '--residual_dir', work],
            'meta': ['bash', 'run_meta.sh', *common, '--save_final_npy', '0', '--residual_dir', work,
                     *self.meta_extra],
        }

    def forecaster_checkpoints(self, cell):
        """Checkpoints my_train.py saved for this cell, oldest first."""
        pattern = os.path.join(glob.escape(self.ckpt_root),
                               f'LatentV20_*_{glob.escape(cell["model_id"])}_{glob.escape(cell["model"])}_*',
                               'checkpoint.pth')
        return sorted(glob.glob(pattern), key=os.path.getmtime)

    def remove_forecaster(self, cell):
        for path in self.forecaster_checkpoints(cell):
            remove(os.path.dirname(path))

    def run_stage(self, cmd, log_path):
        start = time.time()
        with open(log_path, 'w') as log:
            log.write(f'$ {shlex.join(cmd)}\n\n')
            log.flush()
            code = subprocess.call(cmd, cwd=REPO, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        return code, time.time() - start

    def train_metrics(self, setting):
        """my_train.py's own test metrics for this forecaster (computed with the training --step)."""
        try:
            with open(self.train_csv, newline='') as f:
                rows = [row for row in csv.DictReader(f) if row.get('setting') == setting]
        except OSError:
            return {}
        if not rows:
            return {}
        row = rows[-1]
        return {'train_test_mse': to_float(row.get('mse')), 'train_test_mae': to_float(row.get('mae')),
                'train_step': to_int(row.get('step')), 'train_epochs': to_int(row.get('actual_epochs'))}

    @staticmethod
    def meta_metrics(info):
        """Val / test MSE and MAE at alpha = 0, 0.05 and alpha* from my_meta.py's meta_info.json."""
        rows = info['table']  # base (alpha=0), paper (alpha=0.05), chosen on val (alpha*)
        if len(rows) != 3 or rows[0]['alpha'] != 0 or abs(rows[1]['alpha'] - PAPER_ALPHA) > 1e-9:
            raise ValueError('unexpected "table" in meta_info.json')
        metrics = {'alpha_star': info['alpha_star']}
        for (variant, _), row in zip(VARIANTS, rows):
            for split in ('val', 'test'):
                for metric in ('mse', 'mae'):
                    metrics[f'{split}_{metric}_{variant}'] = row[f'{split}_{metric}']
        checks = (info.get('checks') or {}).get('alpha0_vs_base_max_abs_diff') or {}
        history = info.get('history')
        metrics.update(check_alpha0_val=checks.get('val'), check_alpha0_test=checks.get('test'),
                       test_windows=(info.get('windows') or {}).get('test'),
                       meta_epochs=len(history) if history else None, base_setting=info.get('base_setting'))
        return metrics

    def run_cell(self, cell):
        """Train -> residual -> meta for one cell, then keep only the small files. Returns the final status."""
        cell_dir = os.path.join(self.cells_dir, cell['cell'])
        os.makedirs(cell_dir, exist_ok=True)
        status_path = self.status_path(cell)
        logs = {stage: os.path.join(cell_dir, f'{stage}.log') for stage in ('train', 'residual', 'meta')}
        res_dir = os.path.join(self.work, cell['model_id'])
        commands = self.commands(cell)

        st = self.status(cell)
        if (st.get('flags') or {}).get('extra') != self.flags['extra']:
            self.remove_forecaster(cell)  # a forecaster trained with other flags is a different model
            st = {}
        st.update(cell)
        st.update(status='running', stage=None, returncode=None, error=None, flags=self.flags,
                  started_at=now(), finished_at=None, git_commit=self.commit)
        st.setdefault('times', {})
        write_json(status_path, st)

        def fail(stage, code, message=None):
            st.update(status='failed', stage=stage, returncode=code, log=logs[stage], finished_at=now(),
                      error=message or log_tail(logs[stage]))
            if stage == 'train':
                self.remove_forecaster(cell)
                st.update(train_done=False, forecaster_checkpoint=None)
            remove(res_dir)
            write_json(status_path, st)
            return st

        try:
            # 1. LatentTSF forecaster (kept from an earlier attempt when that one finished training)
            ckpt = st.get('forecaster_checkpoint')
            if not (st.get('train_done') and ckpt and os.path.exists(ckpt)):
                self.remove_forecaster(cell)
                st.update(train_done=False, forecaster_checkpoint=None, stage='train')
                write_json(status_path, st)
                code, st['times']['train'] = self.run_stage(commands['train'], logs['train'])
                if code != 0:
                    return fail('train', code)
                found = self.forecaster_checkpoints(cell)
                if not found:
                    return fail('train', code, f'my_train.py saved no checkpoint under {self.ckpt_root}')
                st.update(train_done=True, forecaster_checkpoint=found[-1])
                write_json(status_path, st)

            # 2. latent residual series, without the per-window arrays; drop the CSVs my_meta.py does not read
            remove(res_dir)
            st['stage'] = 'residual'
            write_json(status_path, st)
            code, st['times']['residual'] = self.run_stage(commands['residual'], logs['residual'])
            if code != 0:
                return fail('residual', code)
            for pattern in UNUSED_RESIDUAL_FILES:
                for path in glob.glob(os.path.join(glob.escape(res_dir), pattern)):
                    remove(path)

            # 3. meta-correction, without the decoded test arrays
            st['stage'] = 'meta'
            write_json(status_path, st)
            code, st['times']['meta'] = self.run_stage(commands['meta'], logs['meta'])
            if code != 0:
                return fail('meta', code)
            infos = glob.glob(os.path.join(glob.escape(res_dir), 'meta_*', 'meta_info.json'))
            if len(infos) != 1:
                return fail('meta', code, f'expected one meta_info.json in {res_dir}, found {len(infos)}')
            meta_dir = os.path.dirname(infos[0])

            # 4. collect the metrics and the small files
            try:
                metrics = self.meta_metrics(read_json(infos[0]))
                metrics.update(self.train_metrics(metrics['base_setting']))
                shutil.copy2(infos[0], os.path.join(cell_dir, 'meta_info.json'))
                shutil.copy2(os.path.join(meta_dir, 'alpha_sweep.csv'), os.path.join(cell_dir, 'alpha_sweep.csv'))
                for path in glob.glob(os.path.join(glob.escape(res_dir), '*_info.json')):
                    shutil.copy2(path, os.path.join(cell_dir, 'residual_info.json'))
                if self.args.keep == 'all' and os.path.exists(os.path.join(meta_dir, 'checkpoint.pth')):
                    shutil.move(os.path.join(meta_dir, 'checkpoint.pth'),
                                os.path.join(cell_dir, 'meta_checkpoint.pth'))
            except (OSError, KeyError, TypeError, ValueError) as err:
                return fail('meta', code, f'cannot collect the results in {meta_dir}: {err!r}')

            # 5. clean up: residual CSVs, f_phi checkpoint, the empty folder my_train.py makes for test plots
            remove(res_dir)
            plots = os.path.join(REPO, 'result', metrics['base_setting'] or '')
            if metrics['base_setting'] and os.path.isdir(plots) and not os.listdir(plots):
                os.rmdir(plots)
            if self.args.keep == 'none':
                self.remove_forecaster(cell)
                st['forecaster_checkpoint'] = None
            st.update(status='done', stage=None, metrics=metrics, finished_at=now())
            write_json(status_path, st)
            return st
        except KeyboardInterrupt:
            remove(res_dir)
            st.update(status='interrupted', finished_at=now())
            write_json(status_path, st)
            raise

    # ---- results
    def load_results(self):
        results = {}
        for path in glob.glob(os.path.join(glob.escape(self.cells_dir), '*', 'status.json')):
            st = read_json(path)
            if st and st.get('dataset') in TABLE7_DATASETS and st.get('model') in TABLE7_MODELS:
                results[(st['dataset'], st['model'], int(st['pred_len']))] = st
        return results

    def collect(self):
        """Rewrite results.csv, table7_meta.md and table7_meta.tex from the cell folders."""
        results = self.load_results()
        os.makedirs(self.out, exist_ok=True)
        order = {d: i for i, d in enumerate(TABLE7_DATASETS)}
        models = {m: i for i, m in enumerate(TABLE7_MODELS)}
        keys = sorted(results, key=lambda k: (order[k[0]], models[k[1]], k[2]))
        tmp = os.path.join(self.out, 'results.csv.tmp')
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for key in keys:
                st = results[key]
                row = dict(st.get('metrics') or {})
                times = st.get('times') or {}
                flags = st.get('flags') or {}
                row.update(dataset=st['dataset'], model=st['model'], pred_len=st['pred_len'], status=st.get('status'),
                           failed_stage=st.get('stage') if st.get('status') == 'failed' else None,
                           seconds_train=times.get('train'), seconds_residual=times.get('residual'),
                           seconds_meta=times.get('meta'), extra_flags=flags.get('extra'),
                           meta_extra_flags=flags.get('meta_extra'), finished_at=st.get('finished_at'),
                           forecaster_checkpoint=st.get('forecaster_checkpoint'), git_commit=st.get('git_commit'),
                           error=st.get('error'))
                writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in RESULT_COLUMNS})
        os.replace(tmp, os.path.join(self.out, 'results.csv'))
        done = {k: v['metrics'] for k, v in results.items() if v.get('status') == 'done' and v.get('metrics')}
        with open(os.path.join(self.out, 'table7_meta.md'), 'w') as f:
            f.write(markdown_tables(done))
        with open(os.path.join(self.out, 'table7_meta.tex'), 'w') as f:
            f.write(latex_table(done))
        return results

    # ---- main loop
    def run(self, cells):
        start = time.time()
        limit = self.args.disk_limit_gb - self.args.disk_margin_gb
        states = [self.status(cell).get('status') for cell in cells]
        to_skip_done = sum(s == 'done' for s in states)
        to_skip_failed = 0 if self.args.retry_failed else sum(s == 'failed' for s in states)
        print(f'Table 7 sweep: {len(cells)} cells -> {self.out}')
        print(f'  keep={self.args.keep}  extra={self.flags["extra"] or "none"}  '
              f'meta_extra={self.flags["meta_extra"] or "none"}  WANDB_MODE={self.env["WANDB_MODE"]}')
        print(f'  disk: {disk_used(self.disk_root) / GB:.1f} GB used under {self.disk_root}; '
              f'a new cell starts only below {limit:.1f} GB')
        if to_skip_done or to_skip_failed:
            print(f'  skipping {to_skip_done} finished cell(s)'
                  + (f' and {to_skip_failed} failed cell(s) (rerun them with --retry_failed)' if to_skip_failed else ''))
        ran, failed, stop = 0, [], None
        for index, cell in enumerate(cells, 1):
            state = self.status(cell).get('status')
            if state == 'done' or (state == 'failed' and not self.args.retry_failed):
                continue
            hours = (time.time() - start) / 3600
            if self.args.max_hours is not None and hours >= self.args.max_hours:
                stop = f'time budget reached ({hours:.2f} h of --max_hours {self.args.max_hours})'
                break
            used = disk_used(self.disk_root)
            if used / GB > limit:
                stop = (f'disk limit: {used / GB:.1f} GB used under {self.disk_root}, more than '
                        f'--disk_limit_gb {self.args.disk_limit_gb} minus --disk_margin_gb {self.args.disk_margin_gb}')
                break
            label = f'[{index:{len(str(len(cells)))}d}/{len(cells)}] {cell["dataset"]:<11s} {cell["model"]:<12s} ' \
                    f'{cell["pred_len"]:>3d}'
            print(f'{label}  running ...', flush=True)
            st = self.run_cell(cell)
            ran += 1
            times = '  '.join(f'{stage} {duration(st["times"][stage])}' for stage in ('train', 'residual', 'meta')
                              if stage in st.get('times', {}))
            if st['status'] == 'done':
                m = st['metrics']
                print(f'{label}  {times} | test MSE {m["test_mse_base"]:.4f} -> {m["test_mse_astar"]:.4f} '
                      f'(alpha*={m["alpha_star"]:g}), {m["test_mse_a005"]:.4f} (alpha=0.05) | '
                      f'disk {disk_used(self.disk_root) / GB:.1f} GB', flush=True)
            else:
                failed.append(cell['cell'])
                print(f'{label}  FAILED in {st["stage"]} (exit {st["returncode"]}): {st["error"]}\n'
                      f'{" " * len(label)}  log: {st["log"]}', flush=True)
            self.collect()
        results = self.collect()
        done = sum(st.get('status') == 'done' for st in results.values())
        print(f'\nThis session: {ran} cell(s) run, {len(failed)} failed'
              + (f' ({", ".join(failed)})' if failed else '') + f'; {done} cell(s) done in {self.out} overall.')
        if stop:
            print(f'Stopped early: {stop}. Run the same command again to continue.')
        print(f'Results: {os.path.join(self.out, "results.csv")}, table7_meta.md, table7_meta.tex')
        print(f'Disk used under {self.disk_root}: {disk_used(self.disk_root) / GB:.1f} GB')
        return 1 if failed or (stop and stop.startswith('disk')) else 0

    def dry_run(self, cells):
        print(f'{len(cells)} cells -> {self.out} (run order; WANDB_MODE={self.env["WANDB_MODE"]})')
        for index, cell in enumerate(cells, 1):
            st = self.status(cell)
            state = st.get('status') or 'new'
            if state == 'done':
                action = 'skip (done)'
            elif state == 'failed' and not self.args.retry_failed:
                action = 'skip (failed; --retry_failed reruns it)'
            elif st.get('train_done') and (st.get('flags') or {}).get('extra') == self.flags['extra']:
                action = 'run from the residual step (forecaster already trained)'
            else:
                action = 'run'
            print(f'[{index}/{len(cells)}] {cell["cell"]}: {action}')
            for stage, cmd in self.commands(cell).items():
                print(f'    {stage:<8s} {shlex.join(cmd)}')


# ----------------------------------------------------------------------------------------------- tables
def cell_values(metrics):
    """(MSE, MAE) at alpha = 0, alpha = 0.05 and alpha*, and alpha*; None when the cell is not done."""
    if metrics is None:
        return None
    pairs = [(metrics[f'test_mse_{v}'], metrics[f'test_mae_{v}']) for v, _ in VARIANTS]
    return pairs, metrics['alpha_star']


def average(rows):
    """Mean over the four horizons, as Table 7's Avg. rows; None unless all four are done."""
    if any(row is None for row in rows):
        return None
    pairs = [tuple(sum(row[0][i][j] for row in rows) / len(rows) for j in range(2)) for i in range(len(VARIANTS))]
    return pairs, None


def fmt(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


def dataset_rows(done, dataset, model):
    """Table rows of one dataset and backbone: the four horizons and Avg."""
    rows = [cell_values(done.get((dataset, model, p))) for p in TABLE7_PRED_LENS]
    return list(zip([str(p) for p in TABLE7_PRED_LENS] + ['Avg.'], rows + [average(rows)]))


def markdown_tables(done):
    header = ['# Table 7 with meta-correction', '',
             'Test MSE / MAE (scaled space, every test window) written by `run_table7_meta.py`. '
             '**LatentTSF** is our trained forecaster alone (α = 0); **+Meta** subtracts α·f<sub>φ</sub>(R) in '
             'latent space, with α = 0.05 or α\\* chosen on the validation split of each cell. '
             'The best of the three is in bold. For Traffic and Electricity, `my_train.py` tests with `--step 60`, '
             'so the paper\'s w/LatentTSF numbers use fewer windows than these.', '']
    lines = []
    for model in TABLE7_MODELS:
        datasets = [d for d in TABLE7_DATASETS if any((d, model, p) in done for p in TABLE7_PRED_LENS)]
        if not datasets:
            continue
        lines += [f'## {model}', '',
                  '| Dataset | Horizon | LatentTSF MSE | LatentTSF MAE | +Meta α=0.05 MSE | +Meta α=0.05 MAE '
                  '| +Meta α\\* MSE | +Meta α\\* MAE | α\\* |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for dataset in datasets:
            for i, (horizon, value) in enumerate(dataset_rows(done, dataset, model)):
                name = DISPLAY_NAME.get(dataset, dataset) if i == 0 else ''
                if value is None:
                    lines.append(f'| {name} | {horizon} |' + ' -- |' * 7)
                    continue
                pairs, alpha = value
                columns = []
                for j in range(2):  # MSE, MAE: bold the lowest of the three, as displayed
                    shown = [round(pair[j], 3) for pair in pairs]
                    columns.append([f'**{fmt(s)}**' if s == min(shown) else fmt(s) for s in shown])
                row = [columns[j][i] for i in range(len(VARIANTS)) for j in range(2)]
                lines.append(f'| {name} | {horizon} | ' + ' | '.join(row)
                             + f' | {"--" if alpha is None else f"{alpha:g}"} |')
        lines.append('')
    return '\n'.join(header + (lines or ['No finished cells yet.'])) + '\n'


def latex_table(done):
    groups = (('a005', r'w/Meta ($\alpha{=}0.05$)'), ('astar', r'w/Meta ($\alpha^{*}$)'))
    index = {v: i for i, (v, _) in enumerate(VARIANTS)}
    width = 4 * len(TABLE7_MODELS)
    lines = [
        '% Written by run_table7_meta.py: Table 7 layout with the meta-corrected test MSE / MAE of each backbone.',
        '% alpha* is chosen on the validation split of each cell. Needs \\usepackage{booktabs,multirow,graphicx}.',
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Meta-correction in the latent residual space on the six benchmark datasets of Table 7. '
        r'Test MSE and MAE of each backbone with LatentTSF and the meta-corrector, at $\alpha=0.05$ and at '
        r'$\alpha^{*}$ chosen on the validation split.}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{ll' + 'c' * width + '}',
        r'\toprule',
        r'\multicolumn{2}{c}{Models} & ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{{m}}}' for m in TABLE7_MODELS)
        + r' \\',
        ' '.join(rf'\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}' for i in range(len(TABLE7_MODELS))),
        r'\multicolumn{2}{c}{} & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{title}}}' for model in TABLE7_MODELS
                                                for _, title in groups) + r' \\',
        r'\multicolumn{2}{c}{Metric} & ' + ' & '.join(['MSE & MAE'] * (2 * len(TABLE7_MODELS))) + r' \\',
        r'\midrule',
    ]
    for k, dataset in enumerate(TABLE7_DATASETS):
        rows = {model: dict(dataset_rows(done, dataset, model)) for model in TABLE7_MODELS}
        for i, horizon in enumerate([str(p) for p in TABLE7_PRED_LENS] + ['Avg.']):
            first = rf'\multirow{{5}}{{*}}{{{DISPLAY_NAME.get(dataset, dataset)}}}' if i == 0 else ''
            numbers = []
            for model in TABLE7_MODELS:
                value = rows[model][horizon]
                for variant, _ in groups:
                    pair = None if value is None else value[0][index[variant]]
                    numbers += [fmt(None if pair is None else pair[0]), fmt(None if pair is None else pair[1])]
            lines.append(f'{first} & {horizon} & ' + ' & '.join(numbers) + r' \\')
        lines.append(r'\midrule' if k < len(TABLE7_DATASETS) - 1 else r'\bottomrule')
    lines += [r'\end{tabular}}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------------------------- entry point
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Table 7 experiments (6 datasets x 6 backbones x 4 horizons) through '
                    'run_train.sh -> run_residual.sh -> run_meta.sh, within a limited disk budget.',
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', metavar='NAME',
                        help='subset of ETTm1 ETTm2 ETTh1 ETTh2 traffic electricity (default: all)')
    parser.add_argument('--models', nargs='+', metavar='NAME',
                        help='subset of CMoS DLinear PatchTST TimeBase TimeXer iTransformer (default: all)')
    parser.add_argument('--pred_lens', nargs='+', metavar='P', help='subset of 96 192 336 720 (default: all)')
    parser.add_argument('--out_dir', default='./sweeps/table7_meta',
                        help='results, cell folders and forecaster checkpoints (default: %(default)s)')
    parser.add_argument('--work_dir', default=None,
                        help='where the residual step writes its temporary files (default: <out_dir>/work)')
    parser.add_argument('--keep', choices=['forecaster', 'none', 'all'], default='forecaster',
                        help='checkpoints kept after a cell: forecaster (default), none, or all (also f_phi)')
    parser.add_argument('--extra', default='', metavar='"FLAGS"',
                        help='extra flags for all three scripts, e.g. --extra="--train_epochs 1"')
    parser.add_argument('--meta_extra', default='', metavar='"FLAGS"',
                        help='extra flags for run_meta.sh only, e.g. --meta_extra="--alpha_grid 0,0.05,0.1"')
    parser.add_argument('--num_workers', type=int, default=min(10, os.cpu_count() or 1),
                        help='data loader workers (default: %(default)s, min(10, CPU count))')
    parser.add_argument('--max_hours', type=float, default=None,
                        help='do not start a new cell after this many hours (e.g. 10 for a 12-hour Kaggle session)')
    parser.add_argument('--disk_limit_gb', type=float, default=19.5, help='disk budget (default: %(default)s)')
    parser.add_argument('--disk_margin_gb', type=float, default=3.0,
                        help='room a new cell needs below the budget (default: %(default)s)')
    parser.add_argument('--disk_root', default=None,
                        help='folder whose size counts against the budget '
                             '(default: /kaggle/working when the repo is inside it, else the repo)')
    parser.add_argument('--retry_failed', action='store_true', help='also rerun cells that failed before')
    parser.add_argument('--dry_run', action='store_true', help='print the cells and commands, run nothing')
    parser.add_argument('--collect_only', action='store_true',
                        help='only rebuild results.csv and the tables from <out_dir>/cells')
    return parser.parse_args()


def main():
    args = parse_args()
    sweep = Sweep(args)
    if args.collect_only:
        results = sweep.collect()
        done = sum(st.get('status') == 'done' for st in results.values())
        print(f'{done} done / {len(results)} recorded cell(s) -> {os.path.join(sweep.out, "results.csv")}, '
              f'table7_meta.md, table7_meta.tex')
        return 0
    cells = Sweep.cells(pick(args.datasets, TABLE7_DATASETS, 'dataset'), pick(args.models, TABLE7_MODELS, 'backbone'),
                        pick(args.pred_lens, TABLE7_PRED_LENS, 'horizon'))
    if args.dry_run:
        sweep.dry_run(cells)
        return 0
    try:
        return sweep.run(cells)
    except KeyboardInterrupt:
        sweep.collect()
        print('\nInterrupted: the current cell will restart after its last finished stage next time.')
        return 130


if __name__ == '__main__':
    sys.exit(main())
