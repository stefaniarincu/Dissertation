import numpy as np

kits_ds = [[88.41, 90.09, 91.11, 96.88, 8.19], [88.83, 90.51, 91.74, 96.74, 7.97], [87.53, 89.27, 90.64, 96.47, 8.95]]
lits_ds = [[84.90, 88.42, 89.45, 94.27, 7.83], [84.98, 88.50, 89.59, 94.15, 7.88], [85.64, 89.16, 90.28, 93.89, 7.12]]
lung_ds = [[88.59, 90.98, 91.59, 95.94, 5.69], [88.96, 91.46, 91.56, 96.45, 5.16], [88.72, 91.34, 92.07, 95.76, 5.06]]
ds = [kits_ds, lits_ds, lung_ds]

kits_fused = [[87.98, 89.67, 91.31, 96.19, 8.74], [87.69, 89.46, 91.25, 95.95, 8.75], [87.51, 89.17, 90.49, 96.60, 9.08]]
lits_fused = [[83.81, 87.29, 88.14, 94.52, 9.63], [83.83, 87.39, 88.13, 94.31, 9.08], [84.62, 88.20, 88.98, 93.97, 7.91]]
lung_fused = [[88.61, 90.96, 91.26, 96.34, 5.64], [89.18, 91.67, 92.05, 96.22, 4.99], [88.17, 90.62, 91.75, 95.56, 6.01]]
fused = [kits_fused, lits_fused, lung_fused]

kits_distilled = [[92.11, 93.59, 94.41, 97.12, 4.82], [92.15, 93.73, 94.40, 97.39, 4.60], [92.56, 94.07, 94.68, 97.49, 4.24]]
lits_distilled = [[86.63, 89.91, 90.40, 95.22, 7.02], [86.58, 89.96, 90.76, 94.61, 6.51], [85.94, 89.36, 90.27, 94.37, 7.01]]
lung_distilled = [[89.90, 92.50, 93.77, 95.14, 3.93], [89.93, 92.35, 92.47, 96.57, 4.42], [90.58, 93.04, 93.40, 96.32, 3.72]]
distilled = [kits_distilled, lits_distilled, lung_distilled]

ds_text = 'Baseline '
for i in range(len(ds)):
    mean = np.mean(ds[i], axis=0)
    std = np.std(ds[i], axis=0)
    ds_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(ds_text)

fused_text = 'Fused teacher '
for i in range(len(fused)):
    mean = np.mean(fused[i], axis=0)
    std = np.std(fused[i], axis=0)
    fused_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(fused_text)

distilled_text = 'Student '
for i in range(len(distilled)):
    mean = np.mean(distilled[i], axis=0)
    std = np.std(distilled[i], axis=0)
    distilled_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(distilled_text)

print('\n\n')

ds_text = 'Baseline '
for i in range(len(ds)):
    mean = np.mean(ds[i], axis=0)
    std = np.std(ds[i], axis=0)
    ds_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(ds_text)

fused_text = 'Fused teacher '
for i in range(len(fused)):
    mean = np.mean(fused[i], axis=0)
    std = np.std(fused[i], axis=0)
    fused_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(fused_text)

distilled_text = 'Student '
for i in range(len(distilled)):
    mean = np.mean(distilled[i], axis=0)
    std = np.std(distilled[i], axis=0)
    distilled_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(distilled_text)