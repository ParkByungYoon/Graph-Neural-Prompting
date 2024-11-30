import argparse
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from src.dataset.scene_graph import SceneGraphDataset
from src.model.text_llm import TextLLM
from src.utils.callback import CheckPointCallback

def run(args):
    print(args)
    dataset = SceneGraphDataset(args.data_dir)

    idx_split = dataset.get_idx_split()
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TextLLM(args)

    checkpoint_callback = CheckPointCallback(
        dirpath=f'{args.log_dir}',
        filename=f'{args.model_name}',
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
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
    parser.add_argument('--data_dir', type=str, default='/SSL_NAS/benchmark_data/GQA/')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--max_txt_len', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--llm_model_path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--cache_dir', type=str, default='/SSL_NAS/peoples/bonbak/model')
    parser.add_argument('--model_name', type=str, default='TextNeuralPrompt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Text-Neural-Prompting')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)