import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from skelearn.model_selection import train_test_split

def create_dataloaders(X, y, batch_size=32, test_size=0.2, random_state=42):
    """
    Create DataLoaders for training and testing.

    Args:
        X (np.ndarray): The input features.
        y (np.ndarray): The target labels.
        batch_size (int): The batch size for the DataLoader.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The random seed for reproducibility.
    
    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader