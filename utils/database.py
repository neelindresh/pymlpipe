import os


MODEL_FOLDER_NAME="modelrun"


def create_folder(folder_path,name=None):
    """_summary_:create a folder for storing model information

    Returns:
        str: path for storing model details
    """
    folder=MODEL_FOLDER_NAME
    if name!=None: folder=name
    path=os.path.join(folder_path,folder)
    
    if not os.path.exists(path):
        os.mkdir(path)
    return path