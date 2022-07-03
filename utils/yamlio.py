import yaml
import  os
def read_yaml(path):
    if not os.path.exists(path):
        return []
    with open(path) as file:
        fulllist = yaml.load(file, Loader=yaml.FullLoader)
    return fulllist


def write_to_yaml(path,info):
    with open(os.path.join(path), 'w') as file:
        documents = yaml.dump(info, file)
            