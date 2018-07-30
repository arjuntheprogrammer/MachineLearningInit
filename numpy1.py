import numpy as np
# print (np.__version__)
# L=list(range(10))
# print(L)
# print([str(c) for c in L])
# print([type(item) for item in L])

# # Creating Arrays
# print(np.zeros(10, dtype='int'))

# #creating a 3 row x 5 column matrix
# print(np.ones((3,5), dtype=float))

# #creating a matrix with a predefined value
# print(np.full((3,5), 1.23))


# #create an array with a set sequence
# print(np.arange(0, 20, 2))

# #create an array of even space between the given range of values
# print(np.linspace(0, 1, 5))

# #create a 3x3 array with mean 0 and standard deviation 1 in a given dimension
# print(np.random.normal(0, 1, (3,3)))


# # create an identity matrix
# print(np.eye(3))


# #set a random seed
# np.random.seed(0)

# x1 = np.random.randint(10, size=6) #one dimension
# x2 = np.random.randint(10, size=(3,4)) #two dimension
# x3 = np.random.randint(10, size=(3,4,5)) #three dimension

# print("x3 ndim", x3.ndim)
# print("x3 shape", x3.shape)
# print("x3 size", x3.size)

# x1 = np.array([4,3,4,4,8,4])
# print(x1)


# #You can concatenate two or more arrays at once.
# x = np.array([1, 2, 3])
# y = np.array([3, 2, 1])
# z = [21,21,21]
# print(np.concatenate([x, y,z]))


# #You can also use this function to create 2-dimensional arrays.
# grid = np.array([[1,2,3],[4,5,6]])
# print(np.concatenate([grid,grid]))


# #Using its axis parameter, you can define row-wise or column-wise matrix
# print(np.concatenate([grid,grid],axis=1))


# # you can add an array using np.vstack
# x = np.array([3,4,5])
# grid = np.array([[1,2,3],[17,18,19]])
# print(np.vstack([x,grid]))


# #Similarly, you can add an array using np.hstack
# z = np.array([[9],[9]])
# print(np.hstack([grid,z]))

# #Split
# x = np.arange(10)

# x1,x2,x3, = np.split(x,[3,4,6])
# print (x1,x2,x3)

grid = np.arange(16).reshape((4,4))
print(grid)
upper, lower = np.vsplit(grid, [2])
print (upper, lower)



