import os.path
import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
board = cv2.aruco.CharucoBoard_create(5, 7, 4, 2, aruco_dict)


def calibrate_camera():
    cameraMatrix, distCoeffs = None, None
    n = 10
    listCorners, listIDs = [], []
    for i in range(n):
        img = cv2.flip(cv2.imread('charuco-camera-0000{}.png'.format(i)), 1)
        frame = img.copy()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            frame, aruco_dict)
        res = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
        listCorners.append(res[1])
        listIDs.append(res[2])
        print(i)
        return cv2.aruco.calibrateCameraCharuco(listCorners, listIDs, board, (720, 1280), cameraMatrix, distCoeffs)


#(ret, cameraMatrix, distCoeffs,rotation_vectors, translation_vectors) = cv2.aruco.calibrateCameraCharuco(listCorners,listIDs,board,(720,1280),cameraMatrix,distCoeffs)
(ret, cameraMatrix, distCoeffs, rotation_vectors,
 translation_vectors) = calibrate_camera()


def get_markers(img):
    frame = img.copy()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, aruco_dict)
    return corners, ids, rejectedImgPoints


def make_axis(res1, res2):
    rvec, tvec = None, None
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        res[1], res[2], board, cameraMatrix, distCoeffs, rvec, tvec)
    return cameraMatrix, distCoeffs, rvec, tvec


snap = 0

over = cv2.flip(cv2.imread('imagem.png'), 1)


rows, cols, _ = over.shape

# pos4c #np.float32([[132,125],[230,132],[132,162],[230,170]])
pts_original = np.float32(
    [[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])


cap = cv2.VideoCapture(0)


_, frame = cap.read()
print(frame.shape)
while frame is not None:
    frame = frame[:, ::-1, :]

    frame_view = cv2.flip(frame.copy(), 1)
    frame_plane = frame_view.copy()

    corners, ids, rejectedImgPoints = get_markers(frame_view)

    #corners, ids, rejectedImgPoints = cv2.aruco.interpolateCornersCharuco(frame, aruco_dict)
    if ids is not None:
        # print(ids)
        li = len(ids)
        #print("Quantidade de markers: {}".format(len(ids)))

    try:
        res = cv2.aruco.interpolateCornersCharuco(
            corners, ids, frame_view, board)
        # print(res)
        # cv2.aruco.drawDetectedMarkers(frame2,res[0],res[1])

        pos4c = np.array([res[1][0], res[1][20], res[1][3], res[1][23]])
        id4c = np.array([res[2][0], res[2][20], res[2][3], res[2][23]])

        posPoly = np.array([res[1][20], res[1][0], res[1]
                            [3], res[1][23]], np.int32)[:, 0]
        # print(posPoly)

        # np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows -1]])
        pts_corrigida = pos4c

        imaxis = cv2.aruco.drawDetectedCornersCharuco(frame_view, pos4c, id4c)

        cameraMatrix, distCoeffs, rvec, tvec = make_axis(res[1], res[2])

        imaxis = cv2.aruco.drawAxis(
            frame_view, cameraMatrix, distCoeffs, rvec, tvec, 15)
        # print(cameraMatrix)
        if len(cameraMatrix) == 3:

            M = cv2.getPerspectiveTransform(pts_original, pts_corrigida)

            img_corrigida = cv2.warpPerspective(over, M, (cols, rows))

            mask = img_corrigida == 0  # (255-back)
            temp_frame = frame_plane.copy()
            # (back_i & temp_frame)+img_corrigida
            frame_plane = temp_frame*mask+img_corrigida

            print("Detectado")

    except Exception as e:
        print(e)
    imaxis = cv2.aruco.drawDetectedMarkers(frame_view, corners, ids)

    cv2.imshow('frame', imaxis)
    cv2.imshow('frame_plane', frame_plane)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    _, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
