from model import ChessNet
import torch
import torch.nn as nn
from utils import *
from dataset import *

# TRAINING SETUP
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 256
LEAKY_SLOPE = 0.05
DROPOUT = 0.5

if __name__ == "__main__":
    # load dataset
    samples = read_data_sample("data/grouped_train_samples.pkl")
    ds = ChessGroupedDataset(samples)
    # ds = PieceSelectionDataset(samples)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f'Load data: {len(ds)} samples')

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(
        num_classes=64,
        num_channels=6,
        dropout=DROPOUT,
        activation=nn.LeakyReLU(inplace=True, negative_slope=LEAKY_SLOPE),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    # training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        print(f"EPOCH {epoch + 1}")
        print("_" * 10)
        for idx, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                loss_history.append(loss.item())
                epoch_loss += loss.item()
                print(f"batch {idx}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(dl)}")

    # save model
    file_name = f'grouped_ce_leakyrelu{LEAKY_SLOPE}_{EPOCHS}epc_batchsize{BATCH_SIZE}'
    torch.save(model.state_dict(), f"checkpoint/{file_name}.ckpt")
    print("Model saved")

    # save loss history
    with open(f"log/{file_name}.log", "wb") as f:
        pickle.dump(loss_history, f)
    print("Loss history saved")
