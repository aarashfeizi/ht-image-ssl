import torch
from torch import optim
from tqdm import tqdm
import wandb
from sklearn import metrics


def finetune_model(cfg, model, train_loader, val_loader=None):
    import wandb
    OPTS = {'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW}
    
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = cfg.finetune.max_epochs

    optimizer = OPTS[cfg.finetune.opt](model.parameters(),
                                lr=cfg.finetune.lr,
                                weight_decay=cfg.finetune.weight_decay)
    model.cuda()

    def train_one_step(current_epoch):
        total_loss = 0
        pred_lbls = []
        true_lbls = []
        with tqdm(total=len(train_loader), desc=f'{current_epoch}/{epochs}') as t:
            for idx, batch in enumerate(train_loader, start=1):
                _, X, true_lbl = batch
                X = X[0]
                true_lbl = true_lbl[0]
                X = X.cuda()
                X_pred = model(X)['logits']

                acc = None
                true_lbl = true_lbl.cuda()
                loss = loss_fn(X_pred, true_lbl)
                pred_lbls.extend(X_pred.detach().cpu().numpy().argmax(axis=1))
                true_lbls.extend(true_lbl.detach().cpu().numpy())
                acc = metrics.accuracy_score(y_true=true_lbls, y_pred=pred_lbls)
                
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                postfixes = {f'train_CE_loss': total_loss / idx}
                if acc is not None:
                    postfixes.update({'train_acc': acc})

                t.set_postfix(**postfixes)
                t.update()

        return {'acc': acc, 'loss': total_loss / len(train_loader)}
    
    def val(current_epoch):
        total_loss = 0
        pred_lbls = []
        true_lbls = []
        with tqdm(total=len(val_loader), desc=f'{current_epoch}/{epochs}') as t:
            for idx, batch in enumerate(val_loader, start=1):
                X, targets = batch
                X = X.cuda()
                X_pred = model(X)['logits']

                acc = None
                targets = targets.cuda()
                loss = loss_fn(X_pred, targets)
                pred_lbls.extend(X_pred.detach().cpu().numpy().argmax(axis=1))
                true_lbls.extend(targets.detach().cpu().numpy())
                acc = metrics.accuracy_score(y_true=true_lbls, y_pred=pred_lbls)

                total_loss += loss.item()
                
                postfixes = {f'val_CE_loss': total_loss / idx}

                if acc is not None:
                    postfixes.update({'val_acc': acc})

                t.set_postfix(**postfixes)
                t.update()
                
        return {'acc': acc, 'loss': total_loss / len(val_loader)}
    

    val_data = val(0)
    wandb_dict[f'finetuning/val_CE_loss'] = val_data['loss']
    wandb_dict[f'finetuning/val_acc'] = val_data['acc']

    for epoch in range(1, epochs + 1):
        wandb_dict = {}
        train_data = train_one_step(epoch)
        wandb_dict[f'finetuning/train_CE_loss'] = train_data['loss']
        wandb_dict[f'finetuning/train_acc'] = train_data['acc']

        if val_loader is not None:
            val_data = val(epoch)
            wandb_dict[f'finetuning/val_CE_loss'] = val_data['loss']
            wandb_dict[f'finetuning/val_acc'] = val_data['acc']
        
        wandb.log(wandb_dict)
    
    return model
