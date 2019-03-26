gt_file = open('../../data/test_imglist.txt')
pred_file = open('prediction.txt')

gt_lines = gt_file.readlines()
pred_lines = pred_file.readlines()

corr_count = 0
total_count = 0
for i in range(len(gt_lines)):
    gt_class = int(gt_lines[i].strip().split(' ')[1])
    pred_class = int(pred_lines[i].strip())
    if(gt_class == pred_class):
        corr_count += 1
    total_count += 1

print('Accuracy: %.2f%%'%(corr_count*100.0/total_count))
