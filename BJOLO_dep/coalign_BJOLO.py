import dkist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.visualization import ImageNormalize
from astropy.time import Time
from tqdm import tqdm
import cv2
from scipy.signal import medfilt2d
from skimage.transform import warp
import multiprocessing as mp
import dask
import dask.array as da
import os


class VBIData():
    def __init__(self, data, bad_indices, of_params, of_rebin_size,
                save_dir):
        self.data = data
        self.bad_indices = bad_indices
        self.of_params = of_params
        self.of_rebin_size = of_rebin_size
        self.save_dir = save_dir

        nr, nc = self.data.shape[1:] 
        self.row_coords, self.col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    
    def work(self, ii):
        if not ii in self.bad_indices:
            ii_prev = ii - 1

            while ii_prev in self.bad_indices:
                ii_prev = ii_prev - 1

            ref_img = self.data[ii_prev,:,:]
            new_img = self.data[ii,:,:]

            ref_img_8u = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            new_img_8u = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            of = cv2.calcOpticalFlowFarneback(ref_img_8u, new_img_8u, None, *self.of_params)
            of_x, of_y = of[...,0], of[...,1]
            of_x_filter = medfilt2d(of_x, kernel_size=self.of_rebin_size)
            of_y_filter = medfilt2d(of_y, kernel_size=self.of_rebin_size)
            
            self.data[ii,:,:] = warp(self.data[ii,:,:],
            np.array([self.row_coords + of_y_filter,
            self.col_coords + of_x_filter]), mode='edge')
        else:
            pass

    def calc_shift(self,dataset_name):
        for ii in range(1, self.data.shape[0]):
            self.work(ii)

        np.save(os.path.join(self.save_dir,f"{dataset_name}_coaligned.npy"), self.data)
    
    def generate_preview_video(self, date_obs, dataset_name, title="", fps=15, dpi=300):
        fig, ax = plt.subplots(figsize=(5,5))
        norm = ImageNormalize(vmin=np.nanpercentile(self.data[0,:,:],0.2),
                                vmax=np.nanpercentile(self.data[0,:,:],99.8))

        # img_median = np.nanmedian(self.data_wrap[0,:,:])
        # img_std = np.nanstd(self.data_wrap[0,:,:])
        
        im = ax.imshow(self.data[0,:,:], cmap='gray', norm=norm, origin='lower')
        ax.set_title(f"{title} {date_obs[0].strftime('%Y-%m-%dT%H:%M:%S')}")
        frame_list = [index for index in range(1,self.data.shape[0]) if not index in self.bad_indices]

        anim = animation.FuncAnimation(fig, self.update_fig, frames=frame_list,
                                        fargs=(fig, ax, im, date_obs, title), blit=False)
        anim.save(os.path.join(self.save_dir, f"{dataset_name}_preview.mp4"), fps=fps, dpi=dpi)
         
    def update_fig(self, ii, fig, ax, im, date_obs, title):
        im.set_data(self.data[ii,:,:])
        ax.set_title(f"{title} {date_obs[ii].strftime('%Y-%m-%dT%H:%M:%S')}")
        

if __name__ == "__main__":
    vbi_hbeta_dir = "/cluster/scratch/zhuyin/pid_1_123/BJOLO/"
    vbi_hbeta_dataset = dkist.load_dataset(vbi_hbeta_dir)

    vbi_data_crop = vbi_hbeta_dataset.data[:,128:-128,128:-128]

    vbi_data_rebin = da.mean(vbi_data_crop.reshape(vbi_data_crop.shape[0],
                                                vbi_data_crop.shape[1]//4,
                                                4,
                                                vbi_data_crop.shape[2]//4,
                                                4),
                                                axis=(2,4))

    # vbi_data_rebin = vbi_data_rebin[:30,:,:].compute()

    vbi_data_rebin = vbi_data_rebin[:,:,:].compute()

    of_params = [0.5, 3, 15, 3, 5, 1.2, 0]
    of_rebin_size = 33

    bad_indices = np.where(vbi_hbeta_dataset.headers['ATMOS_R0'].value < 0.08)[0]
    # bad_indices = [281]

    save_dir = "/cluster/scratch/zhuyin/pid_1_123/BJOLO_aligned/"

    date_obs = Time(vbi_hbeta_dataset.headers['DATE-AVG'])

    vbi_data_obj = VBIData(vbi_data_rebin, bad_indices, of_params, of_rebin_size,save_dir)
    vbi_data_obj.calc_shift("BJOLO")
    vbi_data_obj.generate_preview_video(date_obs, "BJOLO")





