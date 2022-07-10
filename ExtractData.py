import cv2
from transform import four_point_transform

def cropSudoku(sudoku, debug=False):

    sudoku_grey = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
    sudoku_blurred = cv2.GaussianBlur(sudoku_grey, (7,7), 1)

    thresh = cv2.adaptiveThreshold(sudoku_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    if debug: cv2.imshow("Sudoku Input", thresh)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    sudokuCnt = None

    for cnt in cnts:
        
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, closed=True)

        if len(approx) == 4:
            sudokuCnt = approx
            break

    if sudokuCnt is None:
        raise Exception(("Could not detect Sudoku. Please use a better image."))

    if debug:
        output = sudoku.copy()
        cv2.drawContours(output, [sudokuCnt], -1, (0,255,0), 2)
        cv2.imshow("Sudoku Detected", output)

    sudokuPuzzle = four_point_transform(sudoku, sudokuCnt.reshape(4,2))
    sudokuPuzzle_grey = four_point_transform(sudoku_grey, sudokuCnt.reshape(4,2))

    if debug: cv2.imshow("Image transform", sudokuPuzzle)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (sudokuPuzzle, sudokuPuzzle_grey)


IMAGE_PATH = r"assets\newspaper_sudoku.jpg"
sudoku = cv2.imread(IMAGE_PATH, 1)

cropSudoku(sudoku, debug=True)

