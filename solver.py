def find_next_cell(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(9):
        for y in range(9):
                if grid[x][y] == 0:
                        return x,y
    return -1,-1 # Sentinel for when the sudoku is solved

def is_valid(grid, i, j, n):
    row_valid = all(n != grid[i][x] for x in range(9))
    if row_valid:
        col_valid = all(n != grid[x][j] for x in range(9))
        if col_valid:
            # Coordinates of top-left corner of sub block
            x_corner, y_corner = 3*int(i/3), 3*int(j/3)
            # Check if n is already in the block
            for x in range(x_corner, x_corner+3):
                for y in range(y_corner, y_corner+3):
                    if grid[x][y] == n:
                        return False
            return True
    return False

def solve_sudoku(grid, i=0, j=0):
    i,j = find_next_cell(grid, i, j)
    if i == -1:
        return True
    for n in range(1,10):
        if is_valid(grid,i,j,n):
            grid[i][j] = n
            if solve_sudoku(grid, i, j):
                return True
            # If the sudoku is not solved, undo last step
            grid[i][j] = 0
    return False
