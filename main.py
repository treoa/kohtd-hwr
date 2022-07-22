import os
from pip import main
import toml

if __name__ == '__main__':
    if not os.path.exists('config.toml'):
        print('config.toml not found.')
    with open('config.toml') as f:
        config = toml.load(f)
    height, width, channels = config['VDIR']['INPUT_SHAPE']
    print(height, width, channels)
