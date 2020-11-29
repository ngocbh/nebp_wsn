import os

def change_testdir(working_dir, test_map):
    for file1 in os.listdir(working_dir):
        print(file1)
        a = os.path.join(working_dir, file1)
        if os.path.isdir(a):
            for file2 in os.listdir(a):
                b = os.path.join(a, file2)
                for key in test_map.keys():
                    if key in file2:
                        new_name = file2.replace(key, test_map[key].replace('NIn', 'temp'))
                        print(file2, ' -> ', new_name)
                        os.rename(b, os.path.join(a, new_name))
                        break

            for file2 in os.listdir(a):
                b = os.path.join(a, file2)
                for key in test_map.values():
                    xkey = key.replace('NIn', 'temp')
                    if xkey in file2:
                        new_name = file2.replace(xkey, key)
                        print(file2, ' -> ', new_name)
                        os.rename(b, os.path.join(a, new_name))
                        break


def change_testfile(working_dir, test_map):
    a = working_dir
    if os.path.isdir(a):
        for file2 in os.listdir(a):
            b = os.path.join(a, file2)
            for key in test_map.keys():
                if key in file2:
                    new_name = file2.replace(key, test_map[key].replace('NIn', 'temp'))
                    print(file2, ' -> ', new_name)
                    os.rename(b, os.path.join(a, new_name))
                    break

        for file2 in os.listdir(a):
            b = os.path.join(a, file2)
            for key in test_map.values():
                xkey = key.replace('NIn', 'temp')
                if xkey in file2:
                    new_name = file2.replace(xkey, key)
                    print(file2, ' -> ', new_name)
                    os.rename(b, os.path.join(a, new_name))
                    break

if __name__ == '__main__':
    test_map = {'NIn7': 'NIn7', 'NIn8': 'NIn13', 'NIn9': 'NIn8', 
                'NIn10': 'NIn14', 'NIn11': 'NIn9', 'NIn12': 'NIn15', 
                'NIn13': 'NIn10', 'NIn14': 'NIn16', 'NIn15': 'NIn11', 
                'NIn16': 'NIn17', 'NIn17': 'NIn12', 'NIn18': 'NIn18'}
    # change_testdir('results/ept_sparsity', test_map)
    # change_testfile('data/ept_sparsity', test_map)
