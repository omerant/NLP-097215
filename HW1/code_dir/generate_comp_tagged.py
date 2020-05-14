import subprocess

# execute comp1
args1 = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1_short.wtag', '--test-path',
         'data/comp1_short.words', '--reg-lambda', '0.7', '--run-all', 'true', '--config', 'base', '--beam', '6',
         '--file', 'comp_m1_311773915.wtag']

comp1 = subprocess.Popen(args1, shell=True)
comp1.wait()


# execute comp2
args2 = ['python', 'run_all.py', '--threshold', '1', '--train-path', 'data/train2.wtag', '--test-path',
         'data/comp2.words', '--reg-lambda', '0.1', '--run-all', 'true', '--config', 'base', '--beam', '6',
         '--file', 'comp_m2_201095510.wtag']

comp2 = subprocess.Popen(args2, shell=True)
comp2.wait()



# 'comp_m1_311773915.wtag'