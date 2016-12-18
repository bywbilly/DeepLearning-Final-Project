import os

train_dir = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/train'
test_dir = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/test'
test_new_dir = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/test_new'

def integrate_trivial(target_dir, dirname, names):
    if dirname == train_dir or dirname == test_dir or dirname == test_new_dir:
        return
    print dirname
    with open(target_dir + '_engy', "a") as f1:
        with open(target_dir + '_f0', "a") as f2:
            for data in names:
                #print dirname + '/' + data
                st, ed = None, None
                with open(dirname + '/' + data, 'r') as ff:
                    if data.find('.engy') != -1:
                        lines = 0
                        for line in ff:
                            lines += 1
                        f1.write(str(lines) + '\n')
                        ff.seek(0)
                        for line in ff:
                            f1.write(line)
                        if dirname[-3:] == 'one':
                            f1.write('1\n')
                        elif dirname[-3:] == 'two':
                            f1.write('2\n')
                        elif dirname[-3:] == 'ree':
                            f1.write('3\n')
                        elif dirname[-3:] == 'our':
                            f1.write('4\n')
                        else:
                            print 'miemiemie???'
                    else:
                        lines = 0
                        for line in ff:
                            lines += 1
                        f2.write(str(lines) + '\n')
                        ff.seek(0)
                        for line in ff:
                            f2.write(line)
                        if dirname[-3:] == 'one':
                            f2.write('1\n')
                        elif dirname[-3:] == 'two':
                            f2.write('2\n')
                        elif dirname[-3:] == 'ree':
                            f2.write('3\n')
                        elif dirname[-3:] == 'our':
                            f2.write('4\n')
                        else:
                            print 'miemiemie???'

ret_train, ret_test, ret_test_new = [], [], []

def integrate_nozero(target_dir, dirname, names):
    if dirname == train_dir or dirname == test_dir or dirname == test_new_dir:
        return
    print dirname

    for data in names:
        if data.find('.f0') != -1:
            with open(dirname + '/' + data, 'r') as f:
                st, ed, flip = 0, 0, 0
                for line in f:
                    tmp = float(line.strip())
                    ed += 1
                    if flip == 0 and tmp == 0:
                        st += 1
                    if flip == 0 and tmp != 0:
                        flip ^= 1
                    if flip == 1 and tmp == 0:
                        break
                label = None
                if dirname[-3:] == 'one':
                    label = 1
                elif dirname[-3:] == 'two':
                    label = 2
                elif dirname[-3:] == 'ree':
                    label = 3
                elif dirname[-3:] == 'our':
                    label = 4
                else:
                    print 'miemiemie???'
                assert(label != None)
                if target_dir == 'train':
                    ret_train.append((dirname + '/' + data, st, ed, label))
                    ret_train.append((dirname + '/' + data[:-3] + '.engy', st, ed, label))
                elif target_dir == 'test':
                    ret_test.append((dirname + '/' + data, st, ed, label))
                    ret_test.append((dirname + '/' + data[:-3] + '.engy', st, ed, label))
                elif target_dir == 'test_new':
                    ret_test_new.append((dirname + '/' + data, st, ed, label))
                    ret_test_new.append((dirname + '/' + data[:-3] + '.engy', st, ed, label))
                else:
                    print '!??'


def write_nozero_data(data, target_dir):
    with open(target_dir + '_f0', 'w') as f1:
        with open(target_dir + '_engy', 'w') as f2:
            for item in data:
                #print item
                with open(item[0], 'r') as ff:
                    if (item[0].find('engy') != -1):
                        f2.write(str(item[2]- item[1]) + '\n')
                        index = -1
                        for line in ff:
                            index += 1
                            if index >= item[1] and index < item[2]:
                                f2.write(line)
                        f2.write(str(item[3]) + '\n')
                    else:
                        f1.write(str(item[2]- item[1]) + '\n')
                        index = -1
                        for line in ff:
                            index += 1
                            if index >= item[1] and index < item[2]:
                                f1.write(line)
                        f1.write(str(item[3]) + '\n')

 
if __name__ == "__main__":
    os.path.walk(train_dir, integrate_trivial, train_dir + '_intergrate')
    os.path.walk(test_dir, integrate_trivial, test_dir + '_intergrate')
    os.path.walk(test_new_dir, integrate_trivial, test_new_dir + '_intergrate')
    #os.path.walk(train_dir, integrate_nozero, 'train')
    #write_nozero_data(ret_train, train_dir + '_intergrate_nozero')
    #os.path.walk(test_dir, integrate_nozero, 'test')
    #write_nozero_data(ret_test, test_dir + '_intergrate_nozero')
    #os.path.walk(test_new_dir, integrate_nozero, 'test_new')
    #write_nozero_data(ret_test_new, test_new_dir + '_intergrate_nozero')
