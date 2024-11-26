def train_model(model, train_dataloader, val_dataloader, train_config):
    """
    Trains the SegNet model.
    """

    device = train_config['device']
    n_epochs = train_config['n_epochs']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']
    batches_per_epoch = train_config['batches_per_epoch']
    lr_decay_factor = train_config['lr_decay_factor']

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, masks) in enumerate(train_dataloader):
            if batch_idx >= batches_per_epoch:
                break

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            
            masks = masks.squeeze(1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / batches_per_epoch
        print(f"Epoch [{epoch + 1}/{n_epochs}] Loss: {avg_loss:.4f}")

        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

               
                masks = masks.squeeze(1)

                loss = criterion(outputs, masks)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

    print("Training finished!")

model = ResSegNet(4,3)
train_model(model, train_dataloader, val_dataloader, train_config)
