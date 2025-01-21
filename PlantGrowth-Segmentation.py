import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import urllib.request
from PIL import Image

def calculateContours(img, img_bin, img_gray):

    # Estructuras para las operaciones morfologicas
    struct_big = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))
    struct_small = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))

    # Incrementamos la imagen binaria y despues al hacer la diferencia con la imagen original conseguimos los bordes
    border = cv.dilate(img_bin, struct_big, iterations=1)
    border = border - cv.erode(img_bin, struct_small)

    pltimagesgray(border, "Borders")

    # Esta funcion calcula valores para cada pixel dependiendo de la distancia del pixel con valor 0 mas cercano (negro)
    dt = cv.distanceTransform(img_bin, cv.DIST_L2, 3)

    pltimagesgray(dt, "DistanceTransform")

    # Normalizamos el resultado
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)

    # Blureamos el resultado para eliminar ruido
    dt = cv.GaussianBlur(dt,(7,7),-1)
    
    pltimagesgray(dt, "Calculo DT Blureado")
    
    #PRIMERA IMAGEN
    # 11, -5 172
    #---------------------------------
    # 19 -5 161
    #---------------------------------
    #31, -7 271
    #---------------------------------
    # 17, -11 135
    #---------------------------------
    # 23, -9 144
    #---------------------------------
    # 31, -7 315

    # Threshold adaptativo para extraer los maximos de el resultado de la distancia transform
    dt = cv.adaptiveThreshold(dt, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -5)

    pltimagesgray(dt, "Threshold de DT")

    # Aplicamos operaciones morfologicas para limpiar el resultado de el threshold y conseguir las estructuras que usaremos para el watershed
    dt = cv.erode(dt,struct_small,iterations = 1)
    dt = cv.dilate(dt,struct_big,iterations = 4)

    pltimagesgray(dt, "Puntos para Watershed")

    # Esta funcion asigna un valor unico a cada elemento de la matriz, asignando asi un color
    lbl, ncc = label(dt)

    pltimageslbl(lbl, "LBL imagen")

    print(ncc)    

    # Añadimos los bordes a los puntos que calculamos antes en la misma imagen
    lbl[border >= 80] = 255

    pltimageslbl(lbl, "LBL imagen")

    lbl = lbl.astype(np.int32)
    
    cv.watershed(img, lbl)

    print("[INFO] {} unique segments found".format(len(np.unique(lbl)) - 1))
    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

def pltimages(inImage, title):
    
    inImage = cv.cvtColor(inImage, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.title(title, loc='center')
    plt.imshow(inImage)
    plt.show()

def pltimageslbl(inImage, title):
    
    plt.figure()
    plt.title(title, loc='center')
    plt.imshow(inImage)
    plt.show()

def pltimagesgray(inImage, title):
    
    plt.figure()
    plt.title(title, loc='center')
    plt.imshow(inImage, cmap='gray')
    plt.show()

def cropImage(inImage):
    
    croppedImages = []
    # PLANTA (0,0)
    crop00 = inImage.crop((270, 165, 540, 435))
    crop00 = np.asarray(crop00)
    # PLANTA (0,1)
    crop01 = inImage.crop((710, 165, 980, 435))
    crop01 = np.asarray(crop01)
    # PLANTA (0,2)
    crop02 = inImage.crop((1160, 165, 1430, 435))
    crop02 = np.asarray(crop02)
    # PLANTA (0,3)
    crop03 = inImage.crop((1610, 165, 1890, 435))
    crop03 = np.asarray(crop03)
    # PLANTA (0,4)
    crop04 = inImage.crop((2050, 165, 2320, 435))
    crop04 = np.asarray(crop04)

    # PLANTA (1,0)
    crop10 = inImage.crop((270, 605, 540, 875))
    crop10 = np.asarray(crop10)
    # SUMAMOS 440 PIXELES
    # PLANTA (0,1)
    crop11 = inImage.crop((710, 605, 980, 875))
    crop11 = np.asarray(crop11)
    # PLANTA (0,2)
    crop12 = inImage.crop((1160, 605, 1430, 875))
    crop12 = np.asarray(crop12)
    # PLANTA (0,3)
    crop13 = inImage.crop((1610, 605, 1890, 875))
    crop13 = np.asarray(crop13)
    # PLANTA (0,4)
    crop14 = inImage.crop((2050, 605, 2320, 875))
    crop14 = np.asarray(crop14)

    # PLANTA (2,0)
    crop20 = inImage.crop((270, 1055, 540, 1325))
    crop20 = np.asarray(crop20)
    # #SUMAMOS 440 PIXELES
    # # PLANTA (2,1)
    crop21 = inImage.crop((710, 1055, 980, 1325))
    crop21 = np.asarray(crop21)
    # # PLANTA (2,2)
    crop22 = inImage.crop((1160, 1055, 1430, 1325))
    crop22 = np.asarray(crop22)
    # # PLANTA (2,3)
    crop23 = inImage.crop((1610, 1055, 1890, 1325))
    crop23 = np.asarray(crop23)
    # # PLANTA (2,4)
    crop24 = inImage.crop((2050, 1055, 2320, 1325))
    crop24 = np.asarray(crop24)

    # PLANTA (3,0)
    crop30 = inImage.crop((270, 1505, 540, 1775))
    crop30 = np.asarray(crop30)
    # #SUMAMOS 440 PIXELES
    # # PLANTA (3,1)
    crop31 = inImage.crop((710, 1505, 980, 1775))
    crop31 = np.asarray(crop31)
    # # PLANTA (3,2)
    crop32 = inImage.crop((1160, 1505, 1430, 1775))
    crop32 = np.asarray(crop32)
    # # PLANTA (3,3)
    crop33 = inImage.crop((1610, 1505, 1890, 1775))
    crop33 = np.asarray(crop33)
    # # PLANTA (3,4)
    crop34 = inImage.crop((2050, 1505, 2320, 1775))
    crop34 = np.asarray(crop34)

    croppedImages.append(crop00)
    croppedImages.append(crop01)
    croppedImages.append(crop02)
    croppedImages.append(crop03)
    croppedImages.append(crop04)
    croppedImages.append(crop10)
    croppedImages.append(crop11)
    croppedImages.append(crop12)
    croppedImages.append(crop13)
    croppedImages.append(crop14)
    croppedImages.append(crop20)
    croppedImages.append(crop21)
    croppedImages.append(crop22)
    croppedImages.append(crop23)
    croppedImages.append(crop24)
    croppedImages.append(crop30)
    croppedImages.append(crop31)
    croppedImages.append(crop32)
    croppedImages.append(crop33)
    croppedImages.append(crop34)

    return croppedImages

def percentGreen(inImage):

    hsvImage = cv.cvtColor(inImage, cv.COLOR_BGR2HSV)

    lowerValues = np.array([29, 89, 70])
    upperValues = np.array([179, 255, 255])
    mask = cv.inRange(hsvImage, lowerValues, upperValues)
    mask = cv.medianBlur(mask, 9)
    kernel = np.ones((12,12),np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    height, width = mask.shape[:2] 
    num_pixels = height * width 
    count_white = cv.countNonZero(mask) 
    percent_white = (count_white/num_pixels) * 100 
    percent_white = round(percent_white,2) 

    return percent_white 

def countPlants(imageList):

    counter = 0
    for i in range (len(croppedImages)):
        percentage = percentGreen(croppedImages[i])
        if percentage > 10:
            counter += 1
    
    return counter

if __name__ == '__main__':    

    img8 = cv.imread('/home/rainor/Escritorio/VA/Practica2/PSI_Tray031_2015-12-27--09-03-57_top.png')
    # print(img8.size)  
    # img8 = cv.imread('/home/rainor/Escritorio/VA/Practica2/PSI_Tray031_2016-01-05--20-50-09_top.png')
    # img8 = cv.imread('/home/rainor/Escritorio/VA/Practica2/PSI_Tray031_2016-01-13--19-48-28_top.png')
    # img8 = cv.imread('/home/rainor/Escritorio/VA/Practica2/PSI_Tray032_2015-12-24--14-11-42_top.png')
    # img8 = cv.imread('/home/rainor/Escritorio/VA/Practica2/PSI_Tray032_2016-01-02--14-10-55_top.png')
    # img8 = cv.imread('/home/rainor/Escritorio/VA/Practica2/PSI_Tray032_2016-01-14--14-06-50_top.png')
    
    # imgPIL = Image.open("PSI_Tray031_2015-12-27--09-03-57_top.png")
    # imgPIL = Image.open("PSI_Tray031_2016-01-05--20-50-09_top.png")
    # imgPIL = Image.open("PSI_Tray031_2016-01-13--19-48-28_top.png")
    # imgPIL = Image.open("PSI_Tray032_2015-12-24--14-11-42_top.png")
    # imgPIL = Image.open("PSI_Tray032_2016-01-02--14-10-55_top.png")
    imgPIL = Image.open("PSI_Tray032_2016-01-14--14-06-50_top.png")

    croppedImages = cropImage(imgPIL)
    print(len(croppedImages))
    
    numberPlants = countPlants(croppedImages)

    print(numberPlants)

    #Sharp Image
    kernelSharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    shape = cv.filter2D(img8, -1, kernelSharp)
    pltimages(shape, "Sharp")

    #Contrast Image
    alpha = 1.5
    beta = 0
    contrast = cv.convertScaleAbs(shape, alpha=alpha, beta=beta)
    pltimages(contrast, "Contrast")

    #Segmentacion de color
    hsv = cv.cvtColor(contrast, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (25, 25, 25), (70, 255, 255))

    pltimagesgray(mask, "Binaria con Ruido")

    #Eliminamos Ruido aplicando un filtro de mediana y despues una apertura
    mask = cv.medianBlur(mask, 9)
    kernel = np.ones((12,12),np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    pltimagesgray(mask, "Binaria sin Ruido")

    #Usando la imagen en binario sin ruido cortamos la imagen en color
    imask = mask > 0
    slicer = np.zeros_like(contrast, np.uint8)
    slicer[imask] = contrast[imask]

    pltimages(slicer, "Imagen cortada")

    # Hacemos la imagen binaria
    img_gray = cv.cvtColor(slicer, cv.COLOR_BGR2GRAY)
    # pltimagesgray(img_gray, "Imagen Blanco y negro de la Cortada")

    #Umbralizacion de la imagen en gris ya cortada
    _, img_bin = cv.threshold(img_gray, 0, 100, cv.THRESH_BINARY)

    pltimagesgray(img_bin, "Umbralizacion de la Cortada")

    # Segmentation
    result = calculateContours(contrast, img_bin, img_gray)

    # Resultado final
    result[result != 255] = 0
    result = cv.dilate(result, None)
    contrast[result == 255] = (0, 0, 255)

    pltimages(result, "Resultado Final")

    # contrast[:,:,2] = np.where(result,255,contrast[:,:,2])
    # pltimages(contrast, "Resultado Final")

    img8[:,:,2] = np.where(result,255,img8[:,:,2])

    cv.imwrite('Resultado watershed.png', img8)
    pltimages(img8, "Resultado Final")