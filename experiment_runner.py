import time
import torch
import torch.nn as nn

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from app.load_data import MyCSVDatasetReader as CSVDataset
from app.train import train_network


def run_experiment(
    dataset_path,
    model_class,
    model_name,
    epochs=3,
    bs=30
):

    dataset = CSVDataset(dataset_path)

    train_id, val_id = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=0
    )

    train_set = Subset(dataset, train_id)
    val_set = Subset(dataset, val_id)

    net = model_class()

    device = torch.device("cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adagrad(
        net.parameters(),
        lr=0.5
    )

    start = time.time()

    results = train_network(
        net=net,
        train_set=train_set,
        val_set=val_set,
        device=device,
        epochs=epochs,
        bs=bs,
        optimizer=optimizer,
        criterion=criterion
    )

    results["training_time"] = time.time() - start

    results["dataset"] = dataset_path.split("/")[-1]

    results["model"] = model_name

    return results
