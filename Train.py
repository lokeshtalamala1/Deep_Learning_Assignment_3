import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from SearchUtils import translate
from Seq2SeqRNN import Seq2SeqRNN
from Seq2SeqAttention import Seq2SeqAttention
from VocabUtils import DakshinaLexiconDataset
from VisualizationUtils import (
    plot_detailed_confusion_matrix,
    plot_attention_heatmaps,
    create_transliteration_grid,
    visualize_word_attention,
)

# Ensure Devanagari font
plt.rcParams['font.family'] = 'Nirmala UI'


def get_dataset_paths() -> dict:
    # Specially for Hindi
    language = 'hi'
    base = Path(f"dakshina_dataset_v1.0/dakshina_dataset_v1.0/{language}/lexicons")
    return {
        'train': base / f"{language}.translit.sampled.train.tsv",
        'dev':   base / f"{language}.translit.sampled.dev.tsv",
        'test':  base / f"{language}.translit.sampled.test.tsv",
    }


def pad_collate(batch):
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    pad = 0
    src_padded = [s + [pad] * (max_src - len(s)) for s in srcs]
    tgt_padded = [t + [pad] * (max_tgt - len(t)) for t in tgts]
    return torch.tensor(src_padded), torch.tensor(tgt_padded)


def init_wandb(args):
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    wandb.run.name = f"{args.model_type}-{args.wandb_project}-e{args.epochs}-bs{args.batch_size}"


def build_model(args, vocab_sizes: tuple) -> nn.Module:
    input_dim, output_dim = vocab_sizes
    params = {
        'input_dim': input_dim,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': output_dim,
        'encoder_rnn_type': args.encoder_rnn_type,
        'decoder_rnn_type': args.decoder_rnn_type,
        'encoder_num_layers': args.encoder_layers,
        'decoder_num_layers': args.decoder_layers,
        'pad_idx': args.pad_idx,
    }
    if args.model_type == 'attention':
        params['dropout'] = args.dropout
        return Seq2SeqAttention(**params)
    return Seq2SeqRNN(**params)


def train_epoch(model, loader, optimizer, criterion, device, clip):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(loader, desc='Train'):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, tgt)
        loss = criterion(logits[:,1:].view(-1, logits.size(-1)), tgt[:,1:].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(loader, desc='Eval'):
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, teacher_forcing_ratio=0)
            loss = criterion(out[:,1:].view(-1, out.size(-1)), tgt[:,1:].view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


def test_and_visualize(model, args, vocabs, device):
    ivocab, ovocab = vocabs
    paths = get_dataset_paths()
    test_ds = DakshinaLexiconDataset(paths['test'], ivocab, ovocab, max_len=args.max_seq_len)
    preds, trues = [], []
    out_dir = Path('predictions')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.model_type}_hin_predictions.csv"

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("src,pred,tgt\n")
        for src_ids, tgt_ids in tqdm(test_ds, desc='Test'):
            src_str = ivocab.decode(src_ids)
            tgt_str = ovocab.decode(tgt_ids)
            pred_str, _ = translate(
                model, src_str, ivocab, ovocab, device,
                decode_method=args.decode_method,
                beam_width=args.beam_width,
                length_penalty_alpha=args.length_penalty_alpha
            )
            f.write(f"{src_str},{pred_str},{tgt_str}\n")
            preds.extend(list(pred_str))
            trues.extend(list(tgt_str))

    plot_detailed_confusion_matrix(trues, preds, f"confusion_hin_{args.model_type}.png")
    if args.model_type == 'attention':
        samples = ["namaste", "bharat", "sundar"]
        plot_attention_heatmaps(model, samples, ivocab, ovocab, device)
        for i, s in enumerate(samples):
            visualize_word_attention(
                model, s, ivocab, ovocab, device,
                save_path=f"hin_attn_{i}.png"
            )


def parse_args():
    parser = argparse.ArgumentParser(description='Hindi Transliteration Trainer')
    parser.add_argument('--model_type', choices=['basic','attention'], default='attention')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=600)
    parser.add_argument('--encoder_rnn_type', choices=['lstm','gru'], default='gru')
    parser.add_argument('--decoder_rnn_type', choices=['lstm','gru'], default='gru')
    parser.add_argument('--encoder_layers', type=int, default=2)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clip', type=float, default=2.0)
    parser.add_argument('--decode_method', choices=['greedy','beam'], default='beam')
    parser.add_argument('--beam_width', type=int, default=8)
    parser.add_argument('--length_penalty_alpha', type=float, default=1.0)
    parser.add_argument('--max_seq_len', type=int, default=60)
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='hindi_translit')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--pad_idx', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = get_dataset_paths()

    train_ds = DakshinaLexiconDataset(paths['train'], max_len=args.max_seq_len)
    ivocab, ovocab = train_ds.get_vocabs()
    dev_ds = DakshinaLexiconDataset(paths['dev'], ivocab, ovocab, max_len=args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(args, (len(ivocab), len(ovocab))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx)

    if args.enable_wandb:
        init_wandb(args)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.clip)
        val_loss = evaluate(model, dev_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} » Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        if args.enable_wandb:
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"best_hin_{args.model_type}.pt")

    model.load_state_dict(torch.load(f"best_hin_{args.model_type}.pt"))
    test_and_visualize(model, args, (ivocab, ovocab), device)

if __name__ == '__main__':
    main()







# The parameters for best accuracy are:

# embSize = 256
# hiddenlayerneurons = 32
# cellType = GRU
# batchSize = 64
# encoderlayers = 5
# decodelayers = 5
# bidirection = yes
# dropout = 0.3
# epochs = 10
# hidden_size = 512
# num_layers = 3
# tf_ratio = 0.2
# learningrate = 0.001
# optimizer = adam
