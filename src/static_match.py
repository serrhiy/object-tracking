import os, cv2, numpy

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)


def preprocess_image(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.bilateralFilter(gray, 5, 25, 25)
    return blured


def detect_image_features(image: cv2.typing.MatLike):
    preprocessed = preprocess_image(image)

    harris = cv2.cornerHarris(preprocessed, 2, 3, 0.04)
    keypoints = numpy.argwhere(harris > 0.01 * harris.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in keypoints]

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors


def match_images(image1, image2):
    kp1, des1 = detect_image_features(image1)
    kp2, des2 = detect_image_features(image2)
    bf = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = bf.match(des1, des2)[::5]
    img3 = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matched", img3)
    cv2.waitKey()


def main():
    google_image_path = os.path.join(RESOURCES_DIR, "google.png")
    bing_image_path = os.path.join(RESOURCES_DIR, "bing.png")

    google_image = cv2.imread(google_image_path)
    bing_image = cv2.imread(bing_image_path)

    match_images(google_image, bing_image)


if __name__ == "__main__":
    main()
