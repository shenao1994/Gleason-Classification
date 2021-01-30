from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import os
from skimage import transform


# 生成数据集
class myDataset(Dataset):
    def __init__(self, t2Image_root, adcImage_root, dwiImage_root):
        super(myDataset, self).__init__()
        # self.image_dirs = image_root
        # self.modal_name = modal_name
        # self.image_list = image_name
        # self.label_file = label_file
        self.files = []
        for t2ImgFile, adcImgFile, dwiImgFile in zip(os.listdir(t2Image_root), os.listdir(adcImage_root),
                                                     os.listdir(dwiImage_root)):
            if t2ImgFile.split('.')[0][-1] != '0':
                t2_image_dir = os.path.join(t2Image_root, t2ImgFile)
                adc_image_dir = os.path.join(adcImage_root, adcImgFile)
                dwi_image_dir = os.path.join(dwiImage_root, dwiImgFile)
                self.files.append({
                    "t2Img": t2_image_dir,
                    "adcImg": adc_image_dir,
                    "dwiImg": dwi_image_dir,
                    "label": int(t2_image_dir.split('.')[0][-1]) - 1
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        t2_img_name = datafiles["t2Img"]
        adc_img_name = datafiles["adcImg"]
        dwi_img_name = datafiles["dwiImg"]
        t2_arr = normalise_img(t2_img_name)
        adc_arr = normalise_img(adc_img_name)
        dwi_arr = normalise_img(dwi_img_name)
        label = datafiles["label"]
        lab_arr = label
        img_arr = np.stack((t2_arr, adc_arr, dwi_arr), axis=0)
        return img_arr, lab_arr, t2_img_name


def normalise_img(img_path):
    image = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(image)
    # img_arr = transform.resize(img_arr, (32, 32))
    # 归一化
    max = np.max(img_arr)
    min = np.min(img_arr)
    img_arr = (img_arr - min) / (max - min)
    return img_arr
