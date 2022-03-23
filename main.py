import sys
import cv2
import math
import struct
from datetime import datetime
import glob
import numpy as np

def face_detector3():
    cap_vid = cv2.VideoCapture('fusek_face_car_01.avi')
    cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_profileface.xml")
    smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_righteye_2splits.xml")
    cv2.namedWindow("Camera")
    while True:
        ret, frame = cap_vid.read()
        frame_paint = frame.copy()
        faces,f_rejectLevels,f_weights = face_cascade.detectMultiScale3(frame,scaleFactor=1.05 ,minNeighbors=3, minSize=(200, 200), maxSize=(600, 600), outputRejectLevels = True)
        profiles,p_rejectLevels,p_weights = profile_cascade.detectMultiScale3(frame,scaleFactor=1.1, minNeighbors=3, minSize=(200, 200), maxSize=(600, 600), outputRejectLevels = True)

        for (one_face, one_rejectLevels, one_weights) in zip(faces,f_rejectLevels,f_weights):
            #print("detection " +str(one_rejectLevels) + " w: " + str(one_weights))
            if one_weights > 3.5:
                cv2.rectangle(frame_paint, one_face, (255,255,255), 10)
                cv2.rectangle(frame_paint, one_face, (0,255,0), 2)
                x, y, w, h = one_face
                crop = frame[y:y + h, x:x + w]
                #crop = cv2.resize(crop, (200, 200))

                smiles = smile_cascade.detectMultiScale(crop, scaleFactor=1.05, minNeighbors=300, minSize=(50, 50), maxSize=(300, 300))
                eyes = eye_cascade.detectMultiScale(crop, scaleFactor=1.05, minNeighbors=30, minSize=(40, 40), maxSize=(150, 150))

                for one_smile in smiles:
                    cv2.rectangle(frame_paint, (x+one_smile[0],y+one_smile[1],one_smile[2],one_smile[3]), (255, 255, 255), 10)
                    cv2.rectangle(frame_paint, (x+one_smile[0],y+one_smile[1],one_smile[2],one_smile[3]), (0, 0, 255), 2)
                for one_eye in eyes:
                    cv2.rectangle(frame_paint, (x+one_eye[0],y+one_eye[1],one_eye[2],one_eye[3]), (255, 255, 255), 10)
                    cv2.rectangle(frame_paint, (x+one_eye[0],y+one_eye[1],one_eye[2],one_eye[3]), (255, 0, 0), 2)
        for (one_profile, one_rejectLevels, one_weights) in zip(profiles,p_rejectLevels,p_weights):
            print("detection " +str(one_rejectLevels) + " w: " + str(one_weights))
            if one_weights > 2.5:
                cv2.rectangle(frame_paint, one_profile, (255, 255, 255), 10)
                cv2.rectangle(frame_paint, one_profile, (150, 150, 0), 2)
                '''x, y, w, h = one_profile
                crop = frame[y:y + h, x:x + w]
                # crop = cv2.resize(crop, (200, 200))
                smiles = smile_cascade.detectMultiScale(crop, scaleFactor=1.1, minNeighbors=300, minSize=(50, 50),
                                                        maxSize=(300, 300))
                eyes = eye_cascade.detectMultiScale(crop, scaleFactor=1.1, minNeighbors=40, minSize=(40, 40),
                                                    maxSize=(150, 150))
                for one_smile in smiles:
                    cv2.rectangle(frame_paint, (x + one_smile[0], y + one_smile[1], one_smile[2], one_smile[3]),
                                  (255, 255, 255), 10)
                    cv2.rectangle(frame_paint, (x + one_smile[0], y + one_smile[1], one_smile[2], one_smile[3]),
                                  (0, 0, 255), 2)
                for one_eye in eyes:
                    cv2.rectangle(frame_paint, (x + one_eye[0], y + one_eye[1], one_eye[2], one_eye[3]),
                                  (255, 255, 255), 10)
                    cv2.rectangle(frame_paint, (x + one_eye[0], y + one_eye[1], one_eye[2], one_eye[3]), (255, 0, 0), 2)'''
        cv2.imshow("Camera",frame_paint)
        if cv2.waitKey(1) == 27:
            break

def face_save():
    cap_vid = cv2.VideoCapture('fusek_face_car_01.avi')
    cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    cv2.namedWindow("Face")
    num_faces = 0
    while True:
        num_faces+=1
        ret, frame = cap_vid.read()
        if not ret:
            break
        frame_paint = frame.copy()
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3,minSize=(200,200),maxSize=(600,600))
        for one_face in faces:
            num_faces += 1
            cv2.rectangle(frame_paint, one_face, (255,255,255), 10)
            cv2.rectangle(frame_paint, one_face, (0,255,0), 2)
            x,y,w,h = one_face
            crop = frame[y:y + h, x:x + w]
            cv2.imshow("face", crop)
            cv2.imwrite("faces/"+str(num_faces)+".png",crop)

        cv2.imshow("frame",frame_paint)
        if cv2.waitKey(1) == 27:
            break
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_list = [img for img in glob.glob("faces/*.png")]
    labels = [0 for i in face_list]
    image_train = []
    for name in face_list:
        img = cv2.imread(name,0)
        img =  cv2.resize(img,(120,120))
        image_train.append(img)
    recognizer.train(image_train,np.array(labels))
    recognizer.write("ja.yaml")

def face_recog():
    cap_vid = cv2.VideoCapture('fusek_face_car_01.avi')
    cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    cv2.namedWindow("Face")
    cv2.namedWindow("Camera")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("ja.yaml")
    while True:
        ret, frame = cap_vid.read()
        frame_paint = frame.copy()
        faces = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=3,minSize=(200,200),maxSize=(600,600))
        for one_face in faces:
            x,y,w,h = one_face
            crop = frame[y:y + h, x:x + w]
            crop_grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_grey = cv2.resize(crop_grey, (120, 120))
            predict = recognizer.predict(crop_grey)
            print(predict)
            if (predict[1] <= 80):
                cv2.rectangle(frame_paint, one_face, (255, 255, 255), 10)
                cv2.rectangle(frame_paint, one_face, (0, 255, 0), 2)
            #cv2.imshow("face", crop)
            #cv2.imwrite("faces/"+str(num_faces)+".png",crop)

        cv2.imshow("Camera",frame_paint)
        if cv2.waitKey(1) == 27:
            break

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, one_c):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def sqdif(image, template):
    final_value = 0
    height, width, _ = image.shape
    new_image = image.copy()
    for i in range(height):
        for j in range(width):
            pixel = []
            for k in range(3):
                image_value = int(image[i, j, k])
                template_value = int(template[i, j, k])
                result = image_value - template_value
                dif = math.ceil(result * result) if math.ceil(result * result) <= 255 else 255
                pixel.append(dif)
            new_image[i, j] = pixel

    return new_image

def train_parking():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    park_list = [img for img in glob.glob("training_images/*.png")]
    labels = [0 for i in park_list]
    image_train = []
    for name in park_list:
        img = cv2.imread(name,0)
        image_train.append(img)
    recognizer.train(image_train,np.array(labels))
    recognizer.write("parking.yaml")

def advanced_parking_recog():
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("parking.yaml")

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    temp = cv2.imread("template.png")
    size = (80, 80)
    temp = cv2.resize(temp, size)
    cv2.namedWindow("image_clone", 0)
    cv2.namedWindow("one_place_img", 0)
    #cv2.namedWindow("temp", 0)
    #cv2.namedWindow("source", 0)
    n_park = 0
    font = cv2.FONT_HERSHEY_PLAIN
    image_idx = 0
    for img_name in test_images:
        txt_name = img_name.replace('jpg', 'txt')

        image = cv2.imread(img_name)
        image_clone = image.copy()
        cv2.imshow("image", image)  # originalni obrazek
        for coord in pkm_coordinates:
            n_park += 1
            temp_val = 0
            # print("coord", coord)
            # ziskani souradnic stredu parkovaciho mista
            pt_1 = (int(coord[0]), int(coord[1]))
            pt_3 = (int(coord[4]), int(coord[5]))
            center = ((pt_1[0] + pt_3[0]) / 2, (pt_1[1] + pt_3[1]) / 2)

            one_place_img = four_point_transform(image, coord)
            one_place_img = cv2.resize(one_place_img, size)
            one_gray = cv2.cvtColor(one_place_img, cv2.COLOR_BGR2GRAY)
            # print("min_val", min_val)
            #cv2.imshow("temp", temp)
            #cv2.imshow("source", val)

            cv2.waitKey(1)
            predict = recognizer.predict(one_gray)
            print(predict)
            if predict[1] <= 80:
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)
                temp_val = 1
            else:
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                temp_val = 0


            cv2.imshow("one_place_img", one_place_img)


        cv2.imshow("image_clone", image_clone)
        cv2.waitKey(1)

def parking_recog():
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    temp = cv2.imread("template.png")
    size = (80, 80)
    temp = cv2.resize(temp, size)
    cv2.namedWindow("image_clone", 0)
    cv2.namedWindow("one_place_img", 0)
    #cv2.namedWindow("temp", 0)
    #cv2.namedWindow("source", 0)
    n_park = 0
    font = cv2.FONT_HERSHEY_PLAIN
    image_idx = 0
    for img_name in test_images:
        value_template = 0
        value_sobel = 0
        value_canny = 0
        counter_template = 0
        counter_canny = 0
        counter_sobel = 0
        txt_name = img_name.replace('jpg', 'txt')
        accuracy_file = open(txt_name, 'r')
        accuracy_score = accuracy_file.readlines()
        accuracy_list = []
        image_accuracy_list = []

        for score in accuracy_score:
            st_line = score.strip()
            accuracy_list.append(int(st_line))

        image = cv2.imread(img_name)
        image_clone = image.copy()
        cv2.imshow("image", image)  # originalni obrazek
        for coord in pkm_coordinates:
            n_park += 1
            image_idx+=1
            temp_val = 0
            # print("coord", coord)
            # ziskani souradnic stredu parkovaciho mista
            pt_1 = (int(coord[0]), int(coord[1]))
            pt_3 = (int(coord[4]), int(coord[5]))
            center = ((pt_1[0] + pt_3[0]) / 2, (pt_1[1] + pt_3[1]) / 2)

            one_place_img = four_point_transform(image, coord)
            one_place_img = cv2.resize(one_place_img, size)
            one_gray = cv2.cvtColor(one_place_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('training_images/' + str(image_idx) + '.png', one_gray)
            source = cv2.matchTemplate(one_place_img, temp, cv2.TM_SQDIFF_NORMED)
            val = sqdif(one_place_img, temp)
            min_val, max_val, min_lock, mac_loc = cv2.minMaxLoc(source)
            # print("min_val", min_val)
            #cv2.imshow("temp", temp)
            #cv2.imshow("source", val)

            cv2.waitKey(1)
            predict = min_val
            if predict > 0.1:
                #cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)
                temp_val = 1
            else:
                #cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                temp_val = 0
            if temp_val == accuracy_list[counter_template]:
                value_template += 1

            cv2.putText(image_clone, str(n_park), (int(center[0]), int(center[1])), font, 3, (255, 255, 255), 2)
            canny_im = cv2.Canny(one_place_img, 250, 500)
            gray = cv2.cvtColor(one_place_img, cv2.COLOR_BGR2GRAY)
            x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1)
            y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1)
            absx = cv2.convertScaleAbs(x)
            absy = cv2.convertScaleAbs(y)
            sobel_im = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

            nonzero_canny = cv2.countNonZero(canny_im)
            nonzero_sobel = np.sum(sobel_im >= 150)

            height, width, _ = one_place_img.shape
            percentage_canny = nonzero_canny/(height*width)*100
            percentage_sobel = nonzero_sobel/(height*width)*100

            print(f'Parking slot: {n_park}')
            print(f'CANNY Size: {height*width}\tNonzero: {nonzero_canny}\tPercentage: {percentage_canny}')
            print(f'SOBEL Size: {height*width}\tNonzero: {nonzero_sobel}\tPercentage: {percentage_sobel}')
            print(f'TEMPLATE MATCH predict: {predict}')
            print(f'______________________________________________________________________________________________')
            if percentage_sobel >= 2.3:
                #cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)
                temp_val = 1
            else:
                #cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                temp_val = 0
            if temp_val == accuracy_list[counter_template]:
                value_sobel += 1

            if percentage_canny >= 2.3:
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)
                temp_val = 1
            else:
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                temp_val = 0
            if temp_val == accuracy_list[counter_template]:
                value_canny += 1
            cv2.imshow("canny", canny_im)
            cv2.imshow("sobel", sobel_im)
            cv2.imshow("one_place_img", one_place_img)
            #cv2.waitKey(50000)
            counter_template += 1
            counter_canny += 1
            counter_sobel += 1

        accuracy_temp = value_template/56*100
        accuracy_sobel = value_sobel/56*100
        accuracy_canny = value_canny/56*100
        print(f'ENDING ACCURACY:')
        print(f'CANNY:\t{accuracy_canny}')
        print(f'SOBEL:\t{accuracy_sobel}')
        print(f'TEMPLATE:\t{accuracy_temp}')
        print(f'______________________________________________________________________________________________')
        print(f'\n\n\n\n\n')
        cv2.imshow("image_clone", image_clone)
        nonzero = cv2.countNonZero(canny_im)

        n_park = 0
        cv2.waitKey(50000)

def main(argv):
    advanced_parking_recog()

if __name__ == "__main__":
    main(sys.argv[1:])
