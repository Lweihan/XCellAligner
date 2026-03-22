from PIL import Image
import os

def convert_png_to_tif_with_dpi(input_path, output_path, process_mode=None, micron_per_pixel=0.5):
    """
    将 PNG 转为 TIF，并写入正确的 DPI 以供 InstanSeg 使用
    :param input_path: 输入 PNG 路径
    :param output_path: 输出 TIF 路径
    :param process_mode: 'L' (灰度) 或 'RGB'
    :param micron_per_pixel: 每个像素代表的微米数 (例如 0.5 代表 20x, 0.25 代表 40x)
    """
    try:
        img = Image.open(input_path)
        
        if process_mode == 'L':
            img = img.convert('L')
        elif process_mode == 'RGB':
            img = img.convert('RGB')
            
        # --- 关键步骤：计算并设置 DPI ---
        # 1 英寸 = 25400 微米
        # DPI = 25400 / (微米/像素)
        dpi_value = 25400 / micron_per_pixel
        
        print(f"设定像素大小: {micron_per_pixel} μm/pixel")
        print(f"计算所得 DPI: {dpi_value:.2f}")

        # 保存时传入 dpi 参数 (x, y)
        img.save(
            output_path, 
            format='TIFF', 
            compression='tiff_lzw',
            dpi=(dpi_value, dpi_value)  # <--- 这里写入了物理尺寸信息
        )
        
        print(f"成功：{input_path} -> {output_path} (已嵌入 DPI)")
        
    except Exception as e:
        print(f"发生错误：{e}")

# 使用示例
# 假设是 20x 物镜 (0.5 μm/pixel)，如果是 40x 请改为 0.25
convert_png_to_tif_with_dpi(
    '/c23227/lwh/dataset/Orion结直肠癌/processed/patch/CRC01/he/channel_1/he_x41984_y38912.png', 
    '/c23227/lwh/dataset/Orion结直肠癌/processed/patch-tif/CRC01/he_x41984_y38912.tif', 
    process_mode='RGB',
    micron_per_pixel=0.25
)