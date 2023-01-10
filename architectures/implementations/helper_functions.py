import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pathlib import Path


# Functionize the training loop
def train(model:nn.Module,
          train_dataloader:DataLoader,
          test_dataloader:DataLoader,
          device:torch.device):
    ### Move model to target device
    model = model.to(device)
    print(f"Moving model to {device} and training on it...")
    
    ### Setting up HyperParameters
    MOMENTUM = 0.9
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    LR_DECAY_FACTOR = 0.1
    EPOCHS = 74

    ##### Create a loss function, optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(params=model.parameters(),
                               lr=LR,
                               weight_decay=WEIGHT_DECAY,
                               momentum=MOMENTUM)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                          factor=LR_DECAY_FACTOR)

    
    # Training the model
    
    train_loss, test_loss = 0, 0
    train_correct, test_correct = 0, 0
    
    print("Training model.....")
    for epoch in tqdm(range(EPOCHS)):

    ### Training loop ###
        model.train()
        for X, y in train_dataloader:

            # Move data to target device
            X, y = X.to(device), y.to(device)

            # Do the forward pass 
            preds = model(X)

            # Calculate the loss
            loss=loss_fn(preds, y)

            # Accumulate the loss
            train_loss+=loss.item()

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()
            
            # Calculate accuracy
            train_correct += (preds.argmax(1) == y).type(torch.float).sum().item()

        ### Testing Loop ###
        model.eval()
        for X, y in test_dataloader:
            with torch.inference_mode():

                # Move data to target device
                X, y = X.to(device), y.to(device)

                # Do the forward pass
                test_pred =  model(X)

                # Calculate the loss
                loss = loss_fn(test_pred, y)
                

                # Accumulate the test loss
                test_loss += loss.item()
                test_correct += (test_pred.argmax(1) == y).type(torch.float).sum().item()
        
        train_correct/=len(train_dataloader.dataset)
        test_correct/=len(test_dataloader.dataset)
        train_loss/=len(train_dataloader)
        test_loss/=len(test_dataloader)
        # Print out what's happening
        print(f"Epoch: {epoch+1} | Train Accuracy: {(100*train_correct):.4f}%  | Train Loss: {train_loss:.8f} | Test Accuracy: {(100*test_correct):.4f}% | Test loss: {test_loss:.8f} \n")
        
def save_model(model, filename):
    # Save the model weights 
    print(f"Saving model to {filename}")
    torch.save(model, f=Path(filename))
