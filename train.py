import argparse
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from src.dataset.obqa import OBQADataset
from src.model.llm import GraphLLM
from src.utils.callback import CheckPointCallback
import os
import wandb

def run(args):
    print(args)
    train_dataset = OBQADataset(
        data_path=os.path.join(args.text_dir, 'train.jsonl'),
        graph_path=os.path.join(args.graph_dir, 'train.graph.adj.pk-nodenum200.loaded_cache'),
        init_emb_path=os.path.join(args.cpnet_dir, 'tzw.ent.npy'),
    )

    valid_dataset = OBQADataset(
        data_path=os.path.join(args.text_dir, 'dev.jsonl'),
        graph_path=os.path.join(args.graph_dir, 'dev.graph.adj.pk-nodenum200.loaded_cache'),
        init_emb_path=os.path.join(args.cpnet_dir, 'tzw.ent.npy'),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = GraphLLM(args)

    checkpoint_callback = CheckPointCallback(
        dirpath=f'{args.log_dir}',
        filename=f'{args.model_name}',
        monitor='valid_total_loss',
        mode='min',
        save_top_k=1
    )

    wandb_logger = pl_loggers.WandbLogger(project=args.wandb_proj, name=args.model_name)
    trainer = pl.Trainer(
        devices=[args.gpu_num],
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print(checkpoint_callback.best_model_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Neural Prompting')

    # General arguments
    parser.add_argument('--cpnet_dir', type=str, default='/SSL_NAS/concrete/data/cpnet')
    parser.add_argument('--text_dir', type=str, default='/SSL_NAS/concrete/data/obqa/statement')
    parser.add_argument('--graph_dir', type=str, default='/SSL_NAS/concrete/data/obqa/graph')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--gpu_num', type=int, default=3)

    # Model specific arguments
    parser.add_argument('--link_drop_rate', type=float, default=0.1)
    parser.add_argument('--lp_lambda', type=float, default=0.1)
    parser.add_argument('--max_txt_len', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--llm_model_path', type=str, default='google/flan-t5-xl')
    parser.add_argument('--cache_dir', type=str, default='/SSL_NAS/peoples/bonbak/model')
    parser.add_argument('--llm_frozen', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default='GNP-LP')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Graph-Neural-Prompting')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)