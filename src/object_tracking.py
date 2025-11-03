import cv2, os, numpy

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)

VIDEO_PATH = os.path.join(RESOURCES_DIR, "stan.mp4")


def mean_shift(capture: cv2.VideoCapture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    ret, frame = capture.read()
    if not ret:
        print("Cannot read frame")
        return

    x, y, w, h = cv2.selectROI("Select ROI", frame, False, False)
    track_window = (x, y, w, h)

    roi = frame[y : y + h, x : x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("MeanShift Tracking", img2)

        k = cv2.waitKey(delay) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def cam_shift(capture: cv2.VideoCapture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    ret, frame = capture.read()
    if not ret:
        print("Cannot read frame")
        return

    x, y, w, h = cv2.selectROI("Select ROI", frame, False, False)
    track_window = (x, y, w, h)

    roi = frame[y : y + h, x : x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("MeanShift Tracking", img2)

        k = cv2.waitKey(delay) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def csrt_tracking(capture: cv2.VideoCapture):
    tracker = cv2.legacy.TrackerCSRT_create()

    ret, frame = capture.read()
    if not ret:
        print("Cannot read video stream")
        return

    bbox = cv2.selectROI("Select ROI", frame, False)
    tracker.init(frame, bbox)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Tracking",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame, "Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        cv2.imshow("CSRT Tracking", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    csrt_tracking(cap)
