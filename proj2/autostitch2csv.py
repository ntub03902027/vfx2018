#! /usr/bin/env python3
import argparse
import os

def createParser():
    parser = argparse.ArgumentParser(description='Convert pano.txt from autostitch to a csv format')
    parser.add_argument('--path', type=str, default='.', metavar='PATH', help='path to the directory where pano.txt locates')
    return parser



if __name__ == '__main__':

    parser = createParser()
    args = parser.parse_args()
	
    
    output = open(os.path.join(args.path, 'pano.csv'), 'w')
    output.write('filename,focal\n')
    filename = []
    focal = []
    with open(os.path.join(args.path, 'pano.txt'), 'r') as infile:
        lines = [line for line in infile]
        for line in lines:
            if '\\' in line:
                line = line.rstrip('\n').split('\\')
                filename.append(line[len(line)-1])

            else:
                line = line.rstrip('\n').rstrip(' ').split(' ')
                if len(line) == 1 and line[0] != '':
                    focal.append(line[0])

    if not len(filename) == len(focal):
        print('Error: parsing not consistent')
        exit(1)

    for i in range(len(filename)):
        output.write('{},{}\n'.format(filename[i], focal[i]))


