import cv2
import numpy as np
import sys
import time

image = cv2.imread("boule.jpg")


#lire la video
image = cv2.VideoCapture(0)   

while(image.isOpened()):
    t1 = time.time()
#lire l'image    
    ret, im = image.read()
    ret, im2 = image.read()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

   
# trouve le bleu dans l'image
    upper_bleu = np.array([120, 255, 255])
    lower_bleu = np.array([100, 50, 50])
    upper_orange = np.array([48, 255, 255])
    lower_orange = np.array([28, 50, 50])

#Detecte les parties bleus
    mask = cv2.inRange(hsv, lower_bleu, upper_bleu)
    mask = cv2.medianBlur(mask,7)
    img = mask.copy()

    edges= cv2.Canny(img, threshold1 = 0, threshold2 = 50, apertureSize = 3)   # detection de contours
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,30,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    if circles != None:                 # Test si il detecte un cercle
        print "balle detectee"
        a, b, c = circles.shape
        for i in range(b):
            cv2.circle(im, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 255, 0), 1, cv2.LINE_AA) # dessine cercle autour
            cv2.circle(im, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv2.LINE_AA) # dessine centre du cercle

    else :
        print "Repartir a la position d'origine"
    t2 = time.time()-t1
    print t2
    cv2.imshow("Image", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
        

#liberer image 
#t2 = time.time()-t1
print t2
image.release()
cv2.destroyAllWindows()
