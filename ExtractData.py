import cv2

IMAGE_PATH = r"assets\newspaper_sudoku.jpg"
sudoku = cv2.imread(IMAGE_PATH, 0)
sudoku_blurred = cv2.GaussianBlur(sudoku, (3,3), 0)

thresh = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow("Sudoku Input", sudoku_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()