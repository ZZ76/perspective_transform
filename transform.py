import cv2
import numpy as np
import _thread

'''
0 initial
1 play with speed estimate
2 points labelling
3 M fix
4 mark and input length
5 measure'''


def mousedisable(event, x, y, flags, param):
    return


def drawpoints(x, y):
    global mask
    if len(pts) == 1:
        cv2.circle(mask, (pts[0][0], pts[0][1]), 2, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.line(mask, (pts[0][0], pts[0][1]), (x, y), (200, 0, 200), thickness=1, lineType=cv2.LINE_AA)
    elif len(pts) < 4:
        pt_last = None
        for pt in pts:
            cv2.circle(mask, (pt[0], pt[1]), 2, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            if pt_last:
                cv2.line(mask, (pt[0], pt[1]), (pt_last[0], pt_last[1]), (200, 0, 200), thickness=1, lineType=cv2.LINE_AA)
                cv2.line(mask, (pts[0][0], pts[0][1]), (pt_last[0], pt_last[1]), (200, 0, 200), thickness=1, lineType=cv2.LINE_AA)
            pt_last = pt
            if pt == pts[-1]:
                cv2.line(mask, (pt[0], pt[1]), (x, y), (200, 0, 200), thickness=1, lineType=cv2.LINE_AA)
                cv2.line(mask, (pts[0][0], pts[0][1]), (x, y), (200, 0, 200), thickness=1, lineType=cv2.LINE_AA)
    if x:
        mask_copy = cv2.copyMakeBorder(mask, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        zoom_window = mask_copy[y:y+100, x:x+100]
        cv2.circle(zoom_window, (50, 50), 2, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        zoom_window = cv2.resize(zoom_window, (200, 200))
        cv2.imshow('zoom', zoom_window)
    return mask


def getperspectiveM():
    global pts, plate_w, plate_h
    print('pts', pts)
    pt1_end = [0, 0]
    pt2_end = [plate_w, 0]
    pt3_end = [plate_w, plate_h]
    pt4_end = [0, plate_h]
    endpts = np.float32([pt1_end, pt2_end, pt3_end, pt4_end])
    pts = np.float32([pts[0], pts[1], pts[2], pts[3]])
    x_min, y_min = np.amin(pts, axis=0)[0], np.amin(pts, axis=0)[1]
    pts_shifted = pts - [x_min, y_min]
    print('start pts:\n', pts.tolist())
    print('endpts:\n', endpts.tolist())
    M = cv2.getPerspectiveTransform(pts_shifted, endpts)
    print('M =\n', M.tolist())
    return M


def getpoints(event, x, y, flags, param):
    global mask, pts, pt_x, pt_y, x_last, y_last
    #print('event', event, 'flags', flags, 'param', param)
    while len(pts) < 4:
        if event == cv2.EVENT_LBUTTONDOWN:  # record start point
            pt_x = x
            pt_y = y
        elif event == cv2.EVENT_LBUTTONUP:  # record end point
            if pt_x:
                pts.append([pt_x, pt_y])
                pt_x, pt_y = None, None
        if x:
            x_last, y_last = x, y
            #print('x, y', x, y, x_last, y_last)
        return mask
    

def getsubtransformedimg():
    global pts, M, mask, outer_pts, inner_pts, offset, size_before_crop, plate_w, plate_h
    #print(pts)
    x_min, x_max = int(np.amin(pts, axis=0)[0]), int(np.amax(pts, axis=0)[0])
    y_min, y_max = int(np.amin(pts, axis=0)[1]), int(np.amax(pts, axis=0)[1])
    outer_pts = [y_min, y_max, x_min, x_max]
    subimg = mask[outer_pts[0]:outer_pts[1], outer_pts[2]:outer_pts[3]]
    subimg = cv2.warpPerspective(subimg, M, (plate_w, plate_h))
    return subimg


def clearall():
    global status, pts, mask, x_last, y_last, M, frame, hiddenmask, marks, scale
    global w_trans, h_trans, outer_pts
    pts, inner_pts, outer_pts, marks = [], [], [], []


def main():
    global status, pts, mask, x_last, y_last, M, frame, hiddenmask, marks, scale
    global w_trans, h_trans, outer_pts, size_before_crop
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('mask')

    x_last, y_last = None, None   # last coord of mouse on window
    status = 0
    pts = []  # clockwise 4 points, start from top-left
    marks = []
    M = None
    scale = None
    outer_pts = []
    size_before_crop = [1000, 1000]

    while True:
        #_, frame = cap.read()
        #frame = cv2.flip(frame, 1)
        #mask = frame.copy()
        mask = cv2.imread('10.jpg')
        mask = cv2.resize(mask, (800, 600))
        h, w, _ = mask.shape
        k = cv2.waitKey(30) & 0xFF  # fps controller

        if status == 0:   # labelling 4 points, get M when ready
            cv2.setMouseCallback('mask', getpoints)
            mask = drawpoints(x_last, y_last)
            if len(pts) == 4:
                M = getperspectiveM()
                #w_trans, h_trans, M = gettransformedsize(M, w, h)  # test
                #pts = []
                status = 1
            if k == ord('l'):
                pts = []

        elif status == 1:  # display transformed window
            cv2.destroyWindow('zoom')
            hiddenmask = getsubtransformedimg()  # test
            cv2.imshow('hiddenmask', hiddenmask)

        if k == ord('q'):
            clearall()
            cv2.destroyAllWindows()
            status = 0
        if k == 27:  # ESC
            break
        #print('status', status)
        cv2.imshow('mask', mask)

#_thread.start_new_thread(main(), ())
#_thread.start_new_thread(getinputlength(), ())
plate_w = 200
plate_h = 200
main()