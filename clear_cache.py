#!/usr/bin/python3
"""Clear the cached results"""
from os import scandir, unlink

if __name__ == '__main__':
    for item in scandir('/dev/shm'):
        if item.name.startswith('cached_'):
            print('Removed file %r' % item.path)
            unlink(item.path)
