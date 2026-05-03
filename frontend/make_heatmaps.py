from PIL import Image, ImageDraw
import os

size = 512
img = Image.new('RGBA', (size, size), (0,0,0,0))
draw = ImageDraw.Draw(img)
center = size // 2
max_radius = 150

for r in range(max_radius, 0, -2):
    alpha = int(255 * (1 - r/max_radius))
    color = (255, max(0, 255 - int(alpha*1.5)), 0, alpha)
    draw.ellipse((center-r, center-r, center+r, center+r), fill=color)

heatmaps = [
    'case_1_effusion.png',
    'case_1_pneumothorax.png',
    'case_2_pulmonary_edema.png',
    'case_3_atelectasis.png',
    'case_3_support_devices.png'
]

for name in heatmaps:
    img.save(f'public/mock-data/heatmaps/{name}')

print('Heatmaps created')
