import subprocess

# execute all flow test1
# print(f'executing flow test1')
# # execute test1
# args_test = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1.wtag', '--test-path',
#          'data/test1.wtag', '--reg-lambda', '0.7', '--run-all', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'test1_our.wtag']
# test1 = subprocess.Popen(args_test, shell=True)
# test1.wait()

#
# execute all flow comp1
# print(f'executing flow comp1')
# args1 = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1_and_test1.wtag', '--test-path',
#          'data/comp1.words', '--reg-lambda', '0.7', '--run-all', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'comp_m1_311773915.wtag']
#
# # comp1 = subprocess.Popen(args1)
# # comp1.wait()
#
# # execute all flow comp2
# print(f'executing flow comp2')
# args2 = ['python', 'run_all.py', '--threshold', '1', '--train-path', 'data/train2.wtag', '--test-path',
#          'data/comp2.words', '--reg-lambda', '0.1', '--run-all', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'comp_m2_201095510.wtag']
#
# comp2 = subprocess.Popen(args2)
# comp2.wait()


# # execute viterbi only test1
# print(f'executing flow test1')
# # execute test1
# args_test = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1.wtag', '--test-path',
#          'data/test1.wtag', '--reg-lambda', '0.7', 'pr', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'test1_our.wtag']
# test1 = subprocess.Popen(args_test, shell=True)
# test1.wait()
#
# execute viterbi only comp1
print(f'executing flow comp1')
args1 = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1_and_test1.wtag', '--test-path',
         'data/comp1.words', '--reg-lambda', '0.7', '--pr', 'true', '--config', 'base', '--beam', '2',
         '--file', 'comp_m1_311773915.wtag']

comp1 = subprocess.Popen(args1)
comp1.wait()

# execute vietrbi only comp2
print(f'executing flow comp2')
args2 = ['python', 'run_all.py', '--threshold', '1', '--train-path', 'data/train2.wtag', '--test-path',
         'data/comp2.words', '--reg-lambda', '0.1', '--pr', 'true', '--config', 'base', '--beam', '2',
         '--file', 'comp_m2_201095510.wtag']

comp2 = subprocess.Popen(args2)
comp2.wait()


# # execute training only test1
# print(f'executing flow test1')
# # execute test1
# args_test = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1.wtag', '--test-path',
#          'data/test1.wtag', '--reg-lambda', '0.7', '--tr', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'test1_our.wtag']
# test1 = subprocess.Popen(args_test, shell=True)
# test1.wait()
#
# # execute training only comp1
# print(f'executing flow comp1')
# args1 = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1_and_test1.wtag', '--test-path',
#          'data/comp1.words', '--reg-lambda', '0.7', '--tr', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'comp_m1_311773915.wtag']
#
# comp1 = subprocess.Popen(args1)
# comp1.wait()
#
# # execute vietrbi only comp2
# print(f'executing flow comp2')
# args2 = ['python', 'run_all.py', '--threshold', '1', '--train-path', 'data/train2.wtag', '--test-path',
#          'data/comp2.words', '--reg-lambda', '0.1', '--tr', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'comp_m2_201095510.wtag']
#
# comp2 = subprocess.Popen(args2)
# comp2.wait()


# # execute pre process only test1
# print(f'executing flow test1')
# # execute test1
# args_test = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1.wtag', '--test-path',
#          'data/test1.wtag', '--reg-lambda', '0.7', '--pp', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'test1_our.wtag']
# test1 = subprocess.Popen(args_test, shell=True)
# test1.wait()
#
# # execute pre process only comp1
# print(f'executing flow comp1')
# args1 = ['python', 'run_all.py', '--threshold', '3', '--train-path', 'data/train1_and_test1.wtag', '--test-path',
#          'data/comp1.words', '--reg-lambda', '0.7', '--pp', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'comp_m1_311773915.wtag']
#
# comp1 = subprocess.Popen(args1)
# comp1.wait()
#
# # execute pre process only comp2
# print(f'executing flow comp2')
# args2 = ['python', 'run_all.py', '--threshold', '1', '--train-path', 'data/train2.wtag', '--test-path',
#          'data/comp2.words', '--reg-lambda', '0.1', '--pp', 'true', '--config', 'base', '--beam', '6',
#          '--file', 'comp_m2_201095510.wtag']
#
# comp2 = subprocess.Popen(args2)
# comp2.wait()
