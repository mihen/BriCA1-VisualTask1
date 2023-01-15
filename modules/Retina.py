import brica1
import cv2
import numpy as np
from cerenaut_pt_core.components.simple_autoencoder import SimpleAutoencoder

from modules.common.obs_convertion import (get_angles_and_image,
                                           get_flatten_observation,
                                           render_image_for_debug)


class Retina(brica1.PipeComponent):
    def _gauss(self, x, sigma):
        sigma_sq = sigma * sigma
        return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x*x/(2 * sigma_sq))

    def _create_rate_datas(self, width, sigma=0.32, clipping_gain=1.2, gain=1.0):
        """ Create mixing rate.
        Arguments:
            width: (int) width of the target image.
            sigma: (float) standard deviation of the gaussian.
            clipping_gain: (float) To make the top of the curve flat, apply gain > 1.0
            gain: (float) Final gain for the mixing rate. 
                          e.g.) if gain=0.8, mixing rates => 0.2~1.0
        Returns:
            Float ndarray (128, 128, 1): Mixing rates and inverted mixing rates. 
        """
        rates = [0.0] * (width * width)
        hw = width // 2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                rates[j*width + i] = self._gauss(r, sigma=sigma)
        rates = np.array(rates)
        # Normalize
        rates = rates / np.max(rates)
        
        # Make top flat by multipying and clipping 
        rates = rates * clipping_gain
        rates = np.clip(rates, 0.0, 1.0)

        # Apply final gain
        if gain != 1.0:
            rates = rates * gain + (1-gain)
        rates = rates.reshape([width, width, 1])
        inv_rates = 1.0 - rates
        return rates, inv_rates

    def _create_blur_image(self, image):
        h = image.shape[0]
        w = image.shape[1]

        # Resizeing to 1/2 size
        resized_image0 = cv2.resize(image,
                                  dsize=(h//2, w//2),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/4 size
        resized_image1 = cv2.resize(resized_image0,
                                  dsize=(h//4, w//4),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/8 size
        resized_image2 = cv2.resize(resized_image1,
                                  dsize=(h//8, w//8),
                                  interpolation=cv2.INTER_LINEAR)
        
        # Resizing to original size
        blur_image = cv2.resize(resized_image2,
                                dsize=(h, w),
                                interpolation=cv2.INTER_LINEAR)

        # Conver to Grayscale
        gray_blur_image = cv2.cvtColor(blur_image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        gray_blur_image = np.reshape(gray_blur_image,
                                     [gray_blur_image.shape[0],
                                      gray_blur_image.shape[0], 1])
        gray_blur_image = np.tile(gray_blur_image, 3)
        return blur_image, gray_blur_image

    def _create_retina_image(self, image):
        blur_image, gray_blur_image = self._create_blur_image(image)
        # Mix original and blur image
        blur_mix_image = image * self.blur_rates + blur_image * self.inv_blur_rates
        # Mix blur mixed image and gray blur image.
        gray_mix_image = blur_mix_image * self.gray_rates + gray_blur_image * self.inv_gray_rates
        return gray_mix_image.astype(np.uint8)

    def __init__(self, observation_dim=1, token_dim=1, config=None):
        super(Retina, self).__init__()
        self.make_in_port('in', observation_dim)
        self.make_out_port('out', observation_dim)
        self.make_in_port('token_in', token_dim)
        self.make_out_port('token_out', token_dim)

        width = 128
        
        self.blur_rates, self.inv_blur_rates = self._create_rate_datas(width)
        self.gray_rates, self.inv_gray_rates = self._create_rate_datas(width, gain=0.5)
        
    def fire(self):
        # フラット化されたベクトルが入力されるので、 Image部とAngle部に分ける
        in_data = self.get_in_port('in').buffer
        (image, angle_h, angle_v) = get_angles_and_image(in_data)

        # Image部をグレースケールにして周辺視野をBlur処理する
        retina_image = self._create_retina_image(image)
        render_image_for_debug(retina_image, "Retina")

        # 再度フラット化して出力する
        self.results['out'] = get_flatten_observation(retina_image, angle_h, angle_v)
        self.results['token_out'] = self.inputs['token_in']
        