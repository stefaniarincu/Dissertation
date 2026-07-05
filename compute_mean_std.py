import numpy as np

'''kits_ds = [[88.41, 90.09, 91.11, 96.88, 8.19], [88.83, 90.51, 91.74, 96.74, 7.97], [87.53, 89.27, 90.64, 96.47, 8.95]]
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
distilled = [kits_distilled, lits_distilled, lung_distilled]'''

bmshare_ds = [[83.13, 85.46, 88.11, 92.82, 11.79], [83.68, 86.03, 88.41, 93.32, 11.04], [83.18, 85.55, 88.46, 92.46, 11.77]]
brats_ds = [[86.93, 89.79, 93.34, 92.86, 7.75], [88.31, 91.11, 93.19, 94.56, 6.91], [87.00, 89.85, 93.53, 92.87, 7.70]]
brats_ped_ds = [[86.90, 89.55, 89.91, 96.32, 8.17], [85.84, 88.38, 88.91, 96.52, 9.64], [86.14, 88.94, 89.05, 96.30, 8.09]]
isles_ds = [[81.85, 85.26, 86.33, 94.64, 10.17], [81.67, 85.32, 86.15, 94.74, 9.44], [82.10, 85.62, 86.51, 94.50, 9.55]]
ds_3 = [bmshare_ds, brats_ds, isles_ds]
ds_4 = [bmshare_ds, brats_ds, brats_ped_ds, isles_ds]

bmshare_fused_3 = [[82.58, 84.59, 85.89, 94.93, 13.44], [82.40, 84.70, 87.09, 93.24, 12.15], [83.28, 85.46, 86.86, 94.18, 11.98]]
brats_fused_3 = [[86.45, 89.48, 92.84, 92.78, 8.09], [86.56, 89.73, 94.04, 91.75, 7.78], [86.99, 90.05, 91.74, 94.59, 7.36]]
isles_fused_3 = [[79.84, 83.20, 83.49, 95.44, 12.15], [79.76, 82.89, 84.02, 94.46, 12.62], [80.27, 83.42, 82.96, 96.51, 12.32]]
fused_3 = [bmshare_fused_3, brats_fused_3, isles_fused_3]

bmshare_fused_4 = [[83.08, 85.31, 87.16, 93.77, 12.15], [83.18, 85.44, 87.36, 93.73, 11.67], [81.77, 83.94, 86.18, 93.22, 13.15]]
brats_fused_4 = [[87.28, 90.22, 92.23, 94.27, 7.48], [87.17, 90.15, 92.31, 94.14, 7.58], [86.26, 89.38, 92.24, 93.11, 8.25]]
brats_ped_fused_4 = [[86.70, 89.43, 89.40, 96.50, 8.11], [85.70, 88.28, 88.41, 96.84, 9.37], [86.03, 88.92, 89.01, 96.01, 7.94]]
isles_fused_4 = [[80.39, 83.54, 84.00, 95.45, 12.37], [80.56, 83.68, 83.07, 96.55, 12.48], [79.65, 82.92, 82.39, 96.16, 12.38]]
fused_4 = [bmshare_fused_4, brats_fused_4, brats_ped_fused_4, isles_fused_4]

bmshare_distilled_3 = [[85.72, 87.99, 89.34, 94.75, 9.48], [85.49, 87.76, 89.33, 94.30, 9.75], [85.71, 88.05, 89.74, 94.22, 9.40]]
brats_distilled_3 = [[88.50, 91.19, 93.07, 94.92, 6.82], [87.80, 90.59, 92.57, 94.68, 7.17], [87.94, 90.82, 92.75, 94.62, 6.61]]
isles_distilled_3 = [[81.92, 85.45, 87.49, 93.56, 9.52], [82.63, 86.24, 87.25, 94.37, 8.71], [82.68, 86.27, 86.26, 95.78, 8.87]]
distilled_3 = [bmshare_distilled_3, brats_distilled_3, isles_distilled_3]

bmshare_distilled_4 = [[85.78, 88.10, 89.70, 94.40, 9.23], [85.63, 87.97, 89.32, 94.47, 9.42], [85.95, 88.25, 89.68, 94.50, 9.22]]
brats_distilled_4 = [[87.42, 90.29, 92.40, 94.32, 7.26], [87.50, 90.37, 92.72, 94.16, 7.50], [86.97, 89.85, 91.84, 94.42, 7.84]]
brats_ped_distilled_4 = [[86.93, 89.68, 90.91, 95.25, 7.96], [86.89, 89.65, 90.12, 95.88, 7.81], [87.03, 89.92, 91.51, 94.87, 7.65]]
isles_distilled_4 = [[82.19, 85.84, 86.99, 94.38, 9.19], [82.51, 86.12, 87.77, 93.87, 9.04], [83.00, 86.59, 87.02, 94.87, 8.53]]
distilled_4 = [bmshare_distilled_4, brats_distilled_4, brats_ped_distilled_4, isles_distilled_4]


ds_text = 'Baseline '
for i in range(len(ds_3)):
    mean = np.mean(ds_3[i], axis=0)
    std = np.std(ds_3[i], axis=0)
    ds_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(ds_text)

fused_text = 'Fused teacher '
for i in range(len(fused_3)):
    mean = np.mean(fused_3[i], axis=0)
    std = np.std(fused_3[i], axis=0)
    fused_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(fused_text)

distilled_text = 'Student '
for i in range(len(distilled_3)):
    mean = np.mean(distilled_3[i], axis=0)
    std = np.std(distilled_3[i], axis=0)
    distilled_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(distilled_text)

ds_text = 'Baseline '
for i in range(len(ds_4)):
    mean = np.mean(ds_4[i], axis=0)
    std = np.std(ds_4[i], axis=0)
    ds_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(ds_text)

fused_text = 'Fused teacher '
for i in range(len(fused_4)):
    mean = np.mean(fused_4[i], axis=0)
    std = np.std(fused_4[i], axis=0)
    fused_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(fused_text)

distilled_text = 'Student '
for i in range(len(distilled_4)):
    mean = np.mean(distilled_4[i], axis=0)
    std = np.std(distilled_4[i], axis=0)
    distilled_text += f'& {mean[0]:.2f} \\pm {std[0]:.2f} & {mean[1]:.2f} \\pm {std[1]:.2f} & {mean[2]:.2f} \\pm {std[2]:.2f} & {mean[3]:.2f} \\pm {std[3]:.2f} & {mean[4]:.2f} \\pm {std[4]:.2f}'
print(distilled_text)