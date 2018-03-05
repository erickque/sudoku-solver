import cv2
from copy import deepcopy

from solver import solve_sudoku
from digit_recognition import extract_grid
from preprocess import preprocess

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Capture',frame)
    key = cv2.waitKey(1)
    if key == 27: # Press escape to quit
        cap.release()
        break
    elif key == 32:
        cap.release()
        
        img = frame
        img = preprocess(img)
        grid = extract_grid(img)
        unsolved = deepcopy(grid)
        solve_sudoku(grid)

        # Writes the unknown numbers onto the board 
        for i in range(9):
            for j in range(9):
                if unsolved[j][i] is 0:
                    cv2.putText(img, str(grid[j][i]),
                                (i*50+15,(j+1)*50-15),
                                0,1,(0,255,0))
        cv2.imshow('Solution',img)
        key = cv2.waitKey(1)
        while key is not 27: # Press escape to quit
            key = cv2.waitKey(1)
        break

cv2.destroyAllWindows()
