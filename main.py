import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('osu_2.jpg')

if img is not None:
    # Conversão HSV e Máscaras Híbridas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Isola os "anéis" azuis das notas
    lower_blue = np.array([100, 100, 40])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Isola o branco dos approach circles
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    

    mask = cv2.bitwise_or(mask_blue, mask_white)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    blurred = cv2.GaussianBlur(mask, (7, 7), 2)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=40, param2=30, minRadius=40, maxRadius=200)
    
    output = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[0] > 400 and i[1] > 100: 
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 3)
                cv2.circle(output, (i[0], i[1]), 4, (0, 0, 255), -1)

    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    
    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title("Máscara Híbrida Limpa", fontsize=16)
    
    axs[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Bot osu! - Detecção \"Perfeita\" (Hough)", fontsize=16)

    axs[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Original", fontsize=16)
    
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Imagem não encontrada.")