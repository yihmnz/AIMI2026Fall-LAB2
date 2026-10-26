"""Grid search and controlled ablations with CSV curves and exact resume checks."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from models import MODEL_NAMES, build_model
from dataio import load_data, digest_file

VERSION = 'eeg-reference-suite-v2'
ACTIVATIONS = ('elu', 'relu', 'leakyrelu', 'gelu')


@dataclass(frozen=True)
class Job:
    stage: str
    model: str
    batch_size: int
    lr: float
    seed: int
    dropout: float | None = None
    activation: str = 'native'

    @property
    def key(self):
        p = 'native' if self.dropout is None else f'{self.dropout:g}'
        return f'bs{self.batch_size}_lr{self.lr:g}_do{p}_act{self.activation}_seed{self.seed}'

    def folder(self, root):
        return root/self.stage/self.model/self.key


def json_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def torch_save(path, data):
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(path)


def csv_save(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast(device, precision):
    if precision == 'fp32':
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def epoch_pass(model, x, y, batch, optimizer, device, args):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(y), device=x.device) if training else torch.arange(len(y), device=x.device)
    total = torch.zeros(3, device=device)
    with torch.set_grad_enabled(training):
        for idx in order.split(batch):
            xb, yb = x[idx].to(device), y[idx].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device, args.precision):
                logits = model(xb)
                ce = nn.functional.cross_entropy(logits.float(), yb)
            penalty = model.regularization_loss()
            loss = ce + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; this run is not a completed result')
            if training:
                loss.backward()
                optimizer.step()
                model.apply_constraints()
            total[0] += ce.detach()*len(idx)
            total[1] += loss.detach()*len(idx)
            total[2] += (logits.argmax(1) == yb).sum()
    ce_sum, objective_sum, correct = total.tolist()
    return ce_sum/len(y), objective_sum/len(y), 100*correct/len(y)


@torch.no_grad()
def prediction_metrics(model, x, y, ids, batch, device, precision):
    model.eval()
    predictions = []
    for xb in x.split(batch):
        with autocast(device, precision):
            predictions.extend(model(xb.to(device)).argmax(1).cpu().tolist())
    pred = np.asarray(predictions)
    truth = y.cpu().numpy()
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (truth, pred), 1)
    f1 = []
    for c in (0, 1):
        tp = cm[c, c]
        fp, fn = cm[:, c].sum()-tp, cm[c, :].sum()-tp
        f1.append(2*tp/max(2*tp+fp+fn, 1))
    metrics = {'selected_test_acc': 100*float((pred == truth).mean()), 'test_macro_f1': 100*float(np.mean(f1))}
    for subject in np.unique(ids):
        mask = ids == subject
        metrics[f'{subject}_test_acc'] = 100*float((pred[mask] == truth[mask]).mean())
    return metrics, cm, [{'trial': i, 'subject': str(ids[i]), 'label': int(truth[i]), 'prediction': int(pred[i])} for i in range(len(y))]


def plot_history(folder, history):
    for metric, label in (('acc', 'Accuracy (%)'), ('loss', 'Cross-entropy loss')):
        fig, ax = plt.subplots(figsize=(8, 4))
        for split in ('train', 'test'):
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history], label=split.capitalize(), lw=1)
        ax.set(xlabel='Epoch', ylabel=label)
        if metric == 'acc':
            ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder/f'{metric}_curve.png', dpi=160)
        plt.close(fig)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def train_one(job, args, root, tensors, test_ids, common):
    folder = job.folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    config = {'job': asdict(job), **common}
    config_file = folder/'config.json'
    if config_file.exists() and json.loads(config_file.read_text(encoding='utf-8')) != config:
        raise RuntimeError(f'Configuration/source/data mismatch: {folder}. Choose a new output directory.')
    json_save(config_file, config)
    summary_file = folder/'run_summary.json'
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        if summary['epochs_completed'] != args.epochs:
            raise RuntimeError(f'Invalid completed result in {folder}')
        print(f'[SKIP] {job.model}/{job.key}', flush=True)
        return summary
    device = torch.device(args.device)
    seed_all(job.seed)
    model = build_model(job.model, dropout=job.dropout, act=job.activation, profile=args.profile).to(device)
    optimizer_cls = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=job.lr, weight_decay=args.weight_decay)
    x, y, xt, yt = tensors
    history, best_acc, best_epoch, start, seconds = [], -1., 0, 1, 0.
    best_state = None
    resume_file = folder/'resume.pt'
    if resume_file.exists():
        # Resume files are local trusted artifacts written by this program.
        checkpoint = torch.load(resume_file, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        start = checkpoint['epoch']+1
        best_acc, best_epoch = checkpoint['best_acc'], checkpoint['best_epoch']
        best_state = checkpoint['best_state']
        seconds = checkpoint['seconds']
        restore_rng(checkpoint['rng'])
        print(f'[RESUME] {job.model}/{job.key} at epoch {start}', flush=True)
    started = time.perf_counter()

    def save_resume():
        # Checkpoint contains the best model as well as the latest model, so a
        # power loss between separate file writes cannot desynchronize them.
        snapshot = {'config': config, 'epoch': history[-1]['epoch'],
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'history': history, 'best_acc': best_acc, 'best_epoch': best_epoch,
                    'best_state': best_state, 'rng': rng_state(),
                    'seconds': seconds+time.perf_counter()-started}
        torch_save(resume_file, snapshot)
        csv_save(folder/'history.csv', history)

    try:
        for epoch in range(start, args.epochs+1):
            tr_loss, tr_obj, tr_acc = epoch_pass(model, x, y, job.batch_size, optimizer, device, args)
            te_loss, te_obj, te_acc = epoch_pass(model, xt, yt, args.eval_batch_size, None, device, args)
            history.append({'epoch': epoch, 'train_loss': tr_loss, 'test_loss': te_loss,
                            'train_objective': tr_obj, 'test_objective': te_obj,
                            'train_acc': tr_acc, 'test_acc': te_acc})
            if te_acc > best_acc:
                best_acc, best_epoch = te_acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % args.log_every == 0 or epoch == args.epochs:
                print(f'{job.model} {job.key} [{epoch}/{args.epochs}] train={tr_acc:.2f}% test={te_acc:.2f}% best={best_acc:.2f}%', flush=True)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_resume()
    except KeyboardInterrupt:
        # Do not snapshot a partial epoch as if it were complete. The last
        # atomic checkpoint is authoritative and replay is at most save_every.
        print(f'Interrupted. Re-run the same command to resume {folder}.', flush=True)
        raise
    seconds += time.perf_counter()-started
    torch_save(folder/'last.pt', {'model': model.state_dict(), 'config': config, 'epoch': args.epochs})
    model.load_state_dict(best_state)
    torch_save(folder/'best.pt', {'model': best_state, 'config': config, 'epoch': best_epoch})
    metrics, cm, predictions = prediction_metrics(model, xt, yt, test_ids, args.eval_batch_size, device, args.precision)
    if abs(metrics['selected_test_acc']-best_acc) > 1e-6:
        raise RuntimeError('Reloaded checkpoint score differs from its recorded peak')
    np.savetxt(folder/'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    csv_save(folder/'predictions.csv', predictions)
    csv_save(folder/'history.csv', history)
    plot_history(folder, history)
    summary = {**asdict(job), 'profile': args.profile, 'best_test_acc': best_acc,
               'best_epoch': best_epoch, 'last_test_acc': history[-1]['test_acc'],
               'last_train_acc': history[-1]['train_acc'], 'epochs_completed': args.epochs,
               'parameters': sum(p.numel() for p in model.parameters()), 'seconds': seconds,
               'selection': 'highest_test_accuracy', 'precision': args.precision,
               'run_id': str(folder.relative_to(root)), **metrics}
    json_save(summary_file, summary)
    if (folder/'error.json').exists():
        (folder/'error.json').unlink()
    print(f'[DONE] {job.model}: {best_acc:.4f}% @ epoch {best_epoch}', flush=True)
    return summary


def summaries(root):
    return [json.loads(p.read_text(encoding='utf-8')) for p in root.glob('*/*/*/run_summary.json')]


def aggregate(rows):
    groups = {}
    keys = ('stage', 'model', 'batch_size', 'lr', 'dropout', 'activation')
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    result = []
    for key, values in groups.items():
        scores = [r['best_test_acc'] for r in values]
        result.append({**dict(zip(keys, key)), 'n_seeds': len(values),
                       'mean_best_test_acc': float(np.mean(scores)),
                       'std_best_test_acc': float(np.std(scores, ddof=1)) if len(scores)>1 else None,
                       'mean_last_test_acc': float(np.mean([r['last_test_acc'] for r in values]))})
    return result


def winners(rows):
    out = {}
    for r in rows:
        if r['model'] not in out or r['mean_best_test_acc'] > out[r['model']]['mean_best_test_acc']:
            out[r['model']] = r
    return [out[k] for k in sorted(out)]


def refresh(root, replot=False):
    rows = summaries(root)
    if not rows:
        return
    rows.sort(key=lambda r: (r['stage'], r['model'], r['run_id']))
    csv_save(root/'all_results.csv', rows)
    agg = aggregate(rows)
    csv_save(root/'configuration_results.csv', agg)
    csv_save(root/'best_by_model.csv', winners(agg))
    grid = [r for r in agg if r['stage'] == 'grid']
    csv_save(root/'grid_results.csv', [r for r in rows if r['stage'] == 'grid'])
    csv_save(root/'grid_best_by_model.csv', winners(grid))
    plotdir = root/'comparison_figures'
    plotdir.mkdir(exist_ok=True)
    for model in MODEL_NAMES:
        subset = [r for r in grid if r['model'] == model]
        if subset:
            bs = sorted({r['batch_size'] for r in subset})
            lr = sorted({r['lr'] for r in subset}, reverse=True)
            mat = np.full((len(bs), len(lr)), np.nan)
            for r in subset:
                mat[bs.index(r['batch_size']), lr.index(r['lr'])] = r['mean_best_test_acc']
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(np.ma.masked_invalid(mat), aspect='auto', vmin=0, vmax=100)
            ax.set(xticks=range(len(lr)), xticklabels=[f'{v:g}' for v in lr],
                   yticks=range(len(bs)), yticklabels=bs, xlabel='Learning rate', ylabel='Batch size', title=model)
            for i in range(len(bs)):
                for j in range(len(lr)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center')
            fig.colorbar(im, ax=ax, label='Mean peak test accuracy (%)')
            fig.tight_layout()
            fig.savefig(plotdir/f'grid_{model}_heatmap.png', dpi=160)
            plt.close(fig)
        for stage, parameter in (('dropout', 'dropout'), ('activation', 'activation')):
            ss = [r for r in agg if r['model'] == model and r['stage'] == stage]
            if not ss:
                continue
            if stage == 'activation' and model in ('SCCNet', 'ShallowConvNet'):
                reference = ss[0]
                baseline = next((r for r in agg if r['model'] == model and r['stage'] == 'dropout'
                    and r['batch_size'] == reference['batch_size'] and r['lr'] == reference['lr']
                    and r['dropout'] == reference['dropout']), None)
                if baseline is not None:
                    ss.append({**baseline, 'stage': 'activation', 'activation': 'native(square)'})
            ss.sort(key=lambda r: str(r[parameter]))
            csv_save(root/f'{model}_{stage}_comparison.csv', ss)
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar([str(r[parameter]) for r in ss], [r['mean_best_test_acc'] for r in ss])
            for bar, r in zip(bars, ss):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{r['mean_best_test_acc']:.2f}", ha='center')
            ax.set(ylim=(0, 105), xlabel=parameter.capitalize(), ylabel='Mean peak test accuracy (%)', title=model)
            fig.tight_layout()
            fig.savefig(plotdir/f'{model}_{stage}.png', dpi=160)
            plt.close(fig)
    best = winners(agg)
    fig, ax = plt.subplots(figsize=(9, 4))
    ix = np.arange(len(best))
    ax.bar(ix-0.2, [r['mean_best_test_acc'] for r in best], width=0.4, label='Mean peak test')
    ax.bar(ix+0.2, [r['mean_last_test_acc'] for r in best], width=0.4, label='Mean final test')
    ax.set(xticks=ix, xticklabels=[r['model'] for r in best], ylim=(0, 100), ylabel='Accuracy (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir/'best_models.png', dpi=160)
    plt.close(fig)
    if replot:
        for f in root.glob('*/*/*/history.csv'):
            with f.open(encoding='utf-8-sig') as handle:
                history = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
            plot_history(f.parent, history)


def choose_reference(root, model, stage, args):
    rows = [r for r in summaries(root) if r['stage'] == stage and r['model'] == model]
    if stage == 'grid':
        expected = [Job(stage, model, b, lr, seed) for b in args.batch_sizes for lr in args.learning_rates for seed in args.seeds]
    else:
        ref = choose_reference(root, model, 'grid', args)
        expected = [Job(stage, model, ref['batch_size'], ref['lr'], seed, dropout=p)
                    for p in args.dropouts for seed in args.seeds]
    ids = {r['run_id'] for r in rows}
    missing = [str(j.folder(root).relative_to(root)) for j in expected if str(j.folder(root).relative_to(root)) not in ids]
    if missing:
        raise RuntimeError(f'{model} {stage} incomplete: {len(missing)} runs missing. Finish this stage first.')
    allowed = {str(j.folder(root).relative_to(root)) for j in expected}
    agg = aggregate([r for r in rows if r['run_id'] in allowed])
    return max(agg, key=lambda r: r['mean_best_test_acc'])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).parent/'data')
    p.add_argument('--output-dir', type=Path, default=Path('outputs_reference_v2'))
    p.add_argument('--models', nargs='+', choices=['all', *MODEL_NAMES], default=['all'])
    p.add_argument('--sweep-models', nargs='+', choices=['all', *MODEL_NAMES], default=['EEGNet'])
    p.add_argument('--stages', nargs='+', choices=['grid', 'dropout', 'activation'], default=['grid', 'dropout', 'activation'])
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[32, 64, 128, 256])
    p.add_argument('--learning-rates', nargs='+', type=float, default=[.01, .001, .0001, .00001])
    p.add_argument('--dropouts', nargs='+', type=float, default=[.1, .3, .5, .7, .9])
    p.add_argument('--activations', nargs='+', choices=ACTIVATIONS, default=list(ACTIVATIONS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--data-on-device', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--profile', choices=['paper', 'reference125'], default='paper')
    p.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--normalization', choices=['none', 'global'], default='none')
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--mode', choices=['train', 'plots'], default='train')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--num-threads', type=int, default=4)
    a = p.parse_args()
    if 'all' in a.models:
        a.models = list(MODEL_NAMES)
    if 'all' in a.sweep_models:
        a.sweep_models = list(MODEL_NAMES)
    for name in ('models', 'sweep_models', 'stages', 'batch_sizes', 'learning_rates', 'dropouts', 'activations', 'seeds'):
        setattr(a, name, list(dict.fromkeys(getattr(a, name))))
    # Sweeps operate only on models selected for this invocation.
    a.sweep_models = [m for m in a.sweep_models if m in a.models]
    if any(v < 1 for v in [a.epochs, a.eval_batch_size, a.save_every, a.log_every, a.num_threads, *a.batch_sizes]):
        p.error('Epochs, batch sizes, intervals and threads must be positive')
    if any(not math.isfinite(v) or v <= 0 for v in a.learning_rates) or not math.isfinite(a.weight_decay) or a.weight_decay < 0:
        p.error('Learning rates must be finite and positive; weight decay must be finite and nonnegative')
    if any(not math.isfinite(v) or v < 0 or v >= 1 for v in a.dropouts):
        p.error('Dropout probabilities must be in [0,1)')
    if any(s < 0 or s >= 2**32 for s in a.seeds):
        p.error('Seeds must be in [0, 2**32)')
    if a.precision == 'bf16' and a.device != 'cuda':
        p.error('bf16 mode requires CUDA')
    return a


def main():
    a = parse_args()
    if a.mode == 'plots':
        refresh(a.output_dir, replot=True)
        return
    grid = [Job('grid', m, b, lr, s) for m in a.models for b in a.batch_sizes for lr in a.learning_rates for s in a.seeds]
    total = (len(grid) if 'grid' in a.stages else 0) + len(a.sweep_models)*len(a.seeds)*(
        (len(a.dropouts) if 'dropout' in a.stages else 0)+(len(a.activations) if 'activation' in a.stages else 0))
    print(f'Plan: {total} runs x {a.epochs} epochs = {total*a.epochs:,} epochs')
    print('Checkpoint and configuration selection use TEST accuracy. This is not an independent final test estimate.')
    if a.dry_run:
        for j in grid if 'grid' in a.stages else []:
            print(j)
        print('Sweep models:', a.sweep_models, '| dropout:', a.dropouts, '| activation:', a.activations)
        return
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a CUDA-enabled PyTorch build and verify the driver.')
    if a.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support bf16')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(a.num_threads)
    torch.use_deterministic_algorithms(a.deterministic)
    torch.backends.cudnn.benchmark = not a.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, y, ids, xt, yt, test_ids, metadata = load_data(a.data_dir, a.normalization)
    root = a.output_dir
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: digest_file(p) for p in Path(__file__).parent.glob('*.py')}
    common = {'implementation': VERSION, 'source_hashes': source_hashes, 'data': metadata,
              'epochs': a.epochs, 'precision': a.precision, 'profile': a.profile,
              'optimizer': a.optimizer, 'weight_decay': a.weight_decay,
              'selection': 'highest_test_accuracy', 'deterministic': a.deterministic,
              'data_on_device': a.data_on_device, 'eval_batch_size': a.eval_batch_size,
              'batch_sizes': a.batch_sizes, 'learning_rates': a.learning_rates,
              'dropouts': a.dropouts, 'activations': a.activations, 'seeds': a.seeds,
              'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'device_name': torch.cuda.get_device_name() if a.device == 'cuda' else 'CPU'}
    manifest = root/'experiment_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != common:
        raise RuntimeError('Output directory belongs to different source/data/settings. Use a new --output-dir.')
    json_save(manifest, common)
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'device': torch.cuda.get_device_name() if a.device == 'cuda' else 'CPU',
                   'platform': platform.platform(), 'command': sys.argv}
    json_save(root/'environment_latest.json', environment)
    print(environment, flush=True)
    tensor_device = a.device if a.data_on_device else 'cpu'
    tensors = (torch.as_tensor(x, device=tensor_device), torch.as_tensor(y, device=tensor_device),
               torch.as_tensor(xt, device=tensor_device), torch.as_tensor(yt, device=tensor_device))
    errors = []

    def run_jobs(jobs):
        for job in jobs:
            try:
                train_one(job, a, root, tensors, test_ids, common)
            except KeyboardInterrupt:
                refresh(root)
                raise
            except Exception as error:
                json_save(job.folder(root)/'error.json', {'job': asdict(job), 'error': repr(error), 'traceback': traceback.format_exc()})
                errors.append(job.key)
                print(f'[FAILED] {job.model}/{job.key}: {error}', flush=True)
                if not a.continue_on_error:
                    refresh(root)
                    raise
            refresh(root)
        if errors:
            raise RuntimeError(f'{len(errors)} run(s) failed; fix errors and resume before selecting sweep references.')

    if 'grid' in a.stages:
        run_jobs(grid)
    for model in a.sweep_models:
        if not set(a.stages) & {'dropout', 'activation'}:
            continue
        reference = choose_reference(root, model, 'grid', a)
        b, lr = reference['batch_size'], reference['lr']
        if 'dropout' in a.stages:
            run_jobs([Job('dropout', model, b, lr, s, p) for p in a.dropouts for s in a.seeds])
        if 'activation' in a.stages:
            dropout = choose_reference(root, model, 'dropout', a)['dropout']
            run_jobs([Job('activation', model, b, lr, s, dropout, act) for act in a.activations for s in a.seeds])
    refresh(root)
    json_save(root/'last_invocation_status.json', {'completed': True, 'planned_runs': total, 'errors': errors})
    print(f'Finished. Summary: {root/"best_by_model.csv"}', flush=True)


if __name__ == '__main__':
    main()
