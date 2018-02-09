cmd_head = 'python3 LipNet_Training_One_Out_Progress.py'

with open("../resource/subjects.txt") as f:
    subjects = f.readlines()
    subjects = [x.strip() for x in subjects]

groups = [2, 6, 7, 8, 9]
progress = [1, 2, 3, 4, 5]

fout = open('batch_run.sh', 'w')

for g in groups:
    for s in subjects:
        for p in progress:
            for i in [1, 2, 3]:
                fout.write(cmd_head + ' ' + str(g) + ' ' + s + ' ' + str(p) + '\n')

fout.close()

