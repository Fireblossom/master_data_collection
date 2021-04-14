"""import os

cmd = "CUDA_VISIBLE_DEVICES=3 python language_model_training.py --data=isear --cuda --embedding --bidirection --tokenize "

for filename in list(os.walk('blog_3_data'))[0][2]:
    save = '--save=lm/isear/balance/' + filename[:-4] + '/ '
    addition = '--addition=blog_3_data/' + filename
    # batch_size = 16
    os.system(cmd + save + addition)"""

import subprocess as sp
import os
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language Model')
parser.add_argument('--data', type=str, 
                    help='location of the data corpus')
parser.add_argument('--model', type=str,
                    help='report interval')
parser.add_argument('--unlabel', type=str,
                    help='report interval')
parser.add_argument('--round', type=str,
                    help='report interval')
parser.add_argument('--label', type=str, default=False, metavar='N', nargs='+',
                    help='vocab level')
args = parser.parse_args()


def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  COMMAND = "nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [(int(x.split()[0]), int(x.split()[2])) for i, x in enumerate(memory_free_info)]
  return memory_free_values


if  __name__ == '__main__':
    processes = [sp.Popen('', shell=True), sp.Popen('', shell=True), sp.Popen('', shell=True), sp.Popen('', shell=True)]
    
    unlabel_files = filter(lambda s: s.endswith('.txt'), os.listdir(args.unlabel))

    if args.label:
        nclass = ' --nclass=' + str(len(args.label)) + ' '
    elif args.data == 'isear':
        nclass = ' --nclass=7 '
    elif args.data == 'tec':
        nclass = ' --nclass=6 '

    try:
        while True:
            # log = open(args.data+'_'+args.model+'_'+'log.txt', 'w+')
            time.sleep(5)
            gpu = get_gpu_memory()
            # print(gpu)
            for i, mem in enumerate(gpu):
                # print(processes[i].poll())
                if type(processes[i].poll()) == int and mem[0] >= 3000 and mem[1] <= 70:
                    unlabel_file = next(unlabel_files)
                    if unlabel_file[-3:] == 'txt':
                        gpu_prefix = 'CUDA_VISIBLE_DEVICES='+ str(i) + ' '

                        if args.label:
                                label = '--label ' + ' '.join(args.label) + ' '
                        else:
                            label = ''

                        if args.model == 'lm':
                            script = 'python language_model_training.py '
                            pretrain = ''
                            unlabel = '--addition='+args.unlabel+'/' + unlabel_file
                            save = '--save=lm/' + args.data + '/round' + args.round + '/' + args.unlabel +'/' + unlabel_file[:-4] + '/ '
                            nclass = ' --epochs=10 '

                        if args.model == 'vat':
                            script = 'python vat_training.py '
                            pretrain = ''
                            unlabel = '--addition=' + args.unlabel + '/' + unlabel_file + ' '
                            if args.label:
                                save = '--save=results/smallset/'+ '_'.join(args.label) + args.data + '/' + args.model + '/' + args.unlabel + '/' + unlabel_file[:-4] + '/ '
                            else:
                                save = '--save=results/'+ args.data + '/round' + args.round + '/' + args.model + '/' + args.unlabel + '/' + unlabel_file[:-4] + '/ '
                            
                        elif args.model == 'sssl':
                            script = 'python classifier_training.py '
                            pretrain = '--pre_train=lm/' + args.data + '/'+args.unlabel+'/' + unlabel_file[:-4] + '/ '
                            unlabel = ''
                            if args.label:
                                save = '--save=results/smallset/'+ '_'.join(args.label) + args.data + '/' + args.model + '/' + args.unlabel + '/' + unlabel_file[:-4] + '/ '
                            else:
                                save = '--save=results/'+ args.data + '/round' + args.round + '/' + args.model + '/' + args.unlabel + '/' + unlabel_file[:-4] + '/ '

                            
                        elif args.model == 'both':
                            script = 'python vat_training.py '
                            pretrain = '--pre_train=lm/' + args.data + '/'+args.unlabel+'/' + unlabel_file[:-4] + '/ '
                            unlabel = '--addition=' + args.unlabel + '/' + unlabel_file + ' '
                            if args.label:
                                save = '--save=results/smallset/'+ '_'.join(args.label) + args.data + '/' + args.model + '/' + args.unlabel + '/' + unlabel_file[:-4] + '/ '
                            else:
                                save = '--save=results/'+ args.data + '/round' + args.round + '/' + args.model + '/' + args.unlabel + '/' + unlabel_file[:-4] + '/ '
                        
                        CMD = gpu_prefix + script + '--cuda --embedding --bidirection --tokenize --data=' + args.data + nclass + save + pretrain + unlabel + label
                        print(CMD)
                        # log.write(CMD+'\n')
                        processes[i] = sp.Popen(CMD, shell=True)
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()