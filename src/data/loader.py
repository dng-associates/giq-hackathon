import pandas as pd
from pathlib import Path


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load data from a specified directory.

    Args:
        data_dir (str): The directory containing the data files.
        
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    
    def load_data(filename: str, data_dir: str ="DATASETS"):
        """
        Load data from a specified file.

        Args:
            filename (str): The name of the file to load.
            data_dir (str): The directory containing the data files. Default is "DATASETS".
        
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        
        file_path = Path(data_dir) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        ## Maybe we will need to change the reading way on future.
        return pd.read_excel(file_path)
    
