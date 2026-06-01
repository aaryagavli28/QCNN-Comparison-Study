import torch
from torch.utils.data import DataLoader
import time
from sklearn.metrics import confusion_matrix, accuracy_score


def train_network(
    net=None,
    train_set=None,
    val_set=None,
    device=None,
    epochs=10,
    bs=20,
    optimizer=None,
    criterion=None
):

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=bs,
        shuffle=True
    )

    net = net.to(device)

    best_val_acc = -1
    best_metrics = None
    best_confusion = None

    for epoch in range(epochs):

        t1 = time.time()

        net.train()

        tr_loss = 0

        y_trues = []
        y_preds = []

        # -------------------------
        # TRAINING
        # -------------------------

        for i, sampled_batch in enumerate(train_loader):

            t2 = time.time()

            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.float()
            y = y.long()

            data = data.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = net(data)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            tr_loss += loss.item()

            y_trues += y.cpu().numpy().tolist()

            y_preds += (
                output.detach()
                .cpu()
                .numpy()
                .argmax(axis=1)
                .tolist()
            )

            print(
                'batch({}):{:.4f}'.format(
                    i,
                    time.time() - t2
                )
            )

        tr_acc = accuracy_score(
            y_trues,
            y_preds
        )

        tr_loss = tr_loss / (i + 1)

        train_cnf = confusion_matrix(
            y_trues,
            y_preds
        )

        print(train_cnf)

        print(
            'Epoch:{}, TR_Loss: {:.4f}, TR_Acc: {:.4f}'.format(
                epoch,
                tr_loss,
                tr_acc
            )
        )

        # -------------------------
        # VALIDATION
        # -------------------------

        net.eval()

        val_loss = 0

        y_trues = []
        y_preds = []

        for i, sampled_batch in enumerate(val_loader):

            data = sampled_batch['feature']
            y = sampled_batch['label'].squeeze()

            data = data.float()
            y = y.long()

            data = data.to(device)
            y = y.to(device)

            with torch.no_grad():

                output = net(data)

            loss = criterion(output, y)

            val_loss += loss.item()

            y_trues += y.cpu().numpy().tolist()

            y_preds += (
                output.detach()
                .cpu()
                .numpy()
                .argmax(axis=1)
                .tolist()
            )

        val_acc = accuracy_score(
            y_trues,
            y_preds
        )

        val_loss = val_loss / (i + 1)

        val_cnf = confusion_matrix(
            y_trues,
            y_preds
        )

        print(val_cnf)

        print(
            'Epoch: {} VAL_Loss: {:.4f}, VAL_Acc: {:.4f}'.format(
                epoch,
                val_loss,
                val_acc
            )
        )

        print(
            'Time for Epoch ({}): {:.4f}'.format(
                epoch,
                time.time() - t1
            )
        )

        # -------------------------
        # SAVE BEST EPOCH
        # -------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_metrics = {
                "train_acc": float(tr_acc),
                "train_loss": float(tr_loss),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss)
            }

            best_confusion = val_cnf.tolist()

    # -------------------------
    # RETURN RESULTS
    # -------------------------

    return {
        "train_acc": best_metrics["train_acc"],
        "train_loss": best_metrics["train_loss"],
        "val_acc": best_metrics["val_acc"],
        "val_loss": best_metrics["val_loss"],
        "confusion_matrix": best_confusion
    }