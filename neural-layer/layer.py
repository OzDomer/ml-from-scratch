input_data = [2, 5, 6]          
weights = [[4, 3, 7],           
           [6, 2, 1]]
bias = [4, 1]                   

def neural_layer (input, weight, bias):
    new_vector = []

    for row in weight:
         dot_product = 0
         for i , w in zip(input, row):
            dot_product += i * w
         new_vector.append(dot_product)
    output =  [first_index + second_index for first_index, second_index in zip(new_vector, bias)]
    return output

def ReLu(vector):
    Relu_vector = []
    for i in vector:
        if i <= 0:
            i = 0
        Relu_vector.append(i)
        
     
    return Relu_vector

vector = function(input_data, weights, bias)
print(ReLu(vector))