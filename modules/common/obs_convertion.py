import cv2
import numpy as np


def get_angles_and_image(observation):
    angle_v = observation[-1]
    obs1 = np.delete(observation, -1)
    angle_h = obs1[-1]
    obs2= np.delete(obs1, -1)
    image = obs2.reshape(128, 128, 3)
    return (image, angle_h, angle_v )

def get_flatten_observation(image, angle_h, angle_v):
    flat_retina_image = np.ravel(image).astype(float)
    flat_retina_image1 = np.append(flat_retina_image, angle_h)
    flat_retina_image2 = np.append(flat_retina_image1, angle_v)
    return flat_retina_image2

def render_image_for_debug(image, title):
    # cv2.putText(image,                       # 図形入力画像
    #         title,              # テキスト文字
    #         (80,400),                  # テキスト配置座標
    #         cv2.FONT_HERSHEY_SIMPLEX,  # フォント
    #         5,                         # 文字サイズ
    #         (255,0,0),                 # カラーチャネル(B,G,R)
    #         9,                         # 文字の太さ
    #         cv2.LINE_AA                # フォント整形
    #        )

    # # 画像を表示
    cv2.imshow(title, image)

def get_angle_state(observation):
    return observation
    obs1 = np.delete(observation, -1)
    obs2= np.delete(obs1, -1)
    image = obs2.reshape(10, 10)
    return image