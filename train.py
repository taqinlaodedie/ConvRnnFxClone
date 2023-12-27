import argparse
import pickle
import torch
from network import ConvRNN, FxDataset, RMSELoss

def main(args):
    model = ConvRNN()

    if torch.cuda.is_available():
        print("Use GPU to train")
        device = "cuda:0"
    else:
        print("Use CPU to train")
        device = "cpu"
    model.to(device)

    ds = lambda x, y: FxDataset(
        torch.from_numpy(x).to(device), 
        torch.from_numpy(y).to(device), 
        window_len=args.sequence_length, 
        batch_size=args.batch_size)
    data = pickle.load(open(args.data, "rb"))
    train_dataset = ds(data["x_train"], data["y_train"])
    valid_dataset = ds(data["x_valid"], data["y_valid"])

    loss_function = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    minimum_loss = 1e6

    for i in range(args.max_epochs):
        print("*********** Epoch {} **************".format(i+1))
        train_loss = model.train_epoch(train_dataset, loss_function, optimizer, device, shuffle=True)
        valid_loss = model.valid_epoch(valid_dataset, loss_function, device)
        print("Train loss {}, Valid loss {}".format(train_loss, valid_loss))
        if valid_loss < minimum_loss:
            minimum_loss = valid_loss
            torch.save(model, "model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=100)

    parser.add_argument("--data", default="data.pickle")
    args = parser.parse_args()
    main(args)