import matplotlib.pyplot as plt
M1 =[[0, -2, 2],
[5,  1, 5],
[1,  4, -1]]

M2=[[0, 1, 2],
[3, 4, 5],
[6, 7, 8]]

T = [[1, 1],
     [0, 1]]



def matrix_multiply(M1, M2):
    result = []
    for j in range(len(M2[0])):       # each column of M2
        result_col = [0] * len(M1)
        for i in range(len(M2)):      # each row of M2 / col of M1
            for k in range(len(M1)):  # each row of M1
                result_col[k] += M1[k][i] * M2[i][j]
        result.append(result_col)
    return result
# print(matrix_multiply(M1, M2))



def draw_grid():
    for x in range(-5, 6):
        plt.plot([x, x], [-5, 5], color='gray', alpha=0.5)
    
    for y in range(-5, 6):
        plt.plot([-5, 5], [y, y], color='gray', alpha=0.5)
        

def draw_transformed_grid():
    for x in range(-5, 6):
        xUpperPoint = matrix_multiply(T, [[x], [5]])
        xLowerPoint = matrix_multiply(T, [[x], [-5]])
        plt.plot([xLowerPoint[0][0], xUpperPoint[0][0]], [xLowerPoint[0][1], xUpperPoint[0][1]],  color='red', alpha=0.5)
    
    for y in range(-5, 6):
        yLeftSide = matrix_multiply(T, [[5], [y]])
        yRightSide = matrix_multiply(T, [[-5], [y]])
        plt.plot([yRightSide[0][0], yLeftSide[0][0]], [yRightSide[0][1], yLeftSide[0][1]], color='red', alpha=0.5)
        






draw_grid()
print(matrix_multiply(T, [[1], [1]]))
draw_transformed_grid()
plt.show()
