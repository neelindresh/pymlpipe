import yaml

def read_yaml(path):
    with open(path) as file:
        fulllist = yaml.load(file, Loader=yaml.FullLoader)
    return fulllist