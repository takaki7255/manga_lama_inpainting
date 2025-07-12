from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation

simple_lama = SimpleLama()

img_path = "./004.jpg"
mask_path = "./004_mask.png"

# 画像を読み込み、RGBに変換（3チャンネル）
image = Image.open(img_path).convert('RGB')

# マスクを読み込み、グレースケールに変換（1チャンネル）
mask = Image.open(mask_path).convert('L')

# マスクをバイナリ化（0または255の値にする）
mask_array = np.array(mask)
# 閾値127で二値化し、255のピクセルがinpaintingされる領域
mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)

# マスク領域に膨張処理を適用
binary_mask = mask_array > 0  # バイナリマスクに変換
dilated_mask = binary_dilation(binary_mask, iterations=3)  # 3回膨張処理
mask_array = (dilated_mask * 255).astype(np.uint8)  # 0-255の値に戻す

mask = Image.fromarray(mask_array)

result = simple_lama(image, mask)
result.save("inpainted1.png")