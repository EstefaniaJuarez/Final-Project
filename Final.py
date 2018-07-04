A = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], [0.3025, 0.36, 0.4225, 0.49, 0.5625, 0.64, 0.7225, 0.81, 0.9025, 1], [0.166375, 0.216, 0.274625, 0.343, 0.421875, 0.512, 0.614125, 0.729, 0.857375, 1]]

y = [1.102, 1.099, 1.017, 1.111, 1.117, 1.152, 1.265, 1.380, 1.575, 1.857]

#Scalar Check
def Check_Scalar(scalar):
  status = True
  if (type(scalar) != int) and (type(scalar) != float) and (type(scalar) != complex):
    status = False
  return status

#Vector Check
def Check_Vector(vector):
  status = True
  for i in range(len(vector)):
    if Check_Scalar(vector[i]) == False:
      status = False
  return status

#Matrix Check
def Check_Matrix(matrix):
  status = True
  for i in range(len(matrix)):
    if Check_Vector(matrix[i]) == False:
      status = False
  return status

def twoNorm(vector):
  '''
  twoNorm takes a vector as it's argument. It then computes the sum of  the squares of each element of the vector. It then returns the square root of this sum.
  '''
  # This variable will keep track of the validity of our input.
  inputStatus = True  
  # This for loop will check each element of the vector to see if it's a number. 
  for i in range(len(vector)):  
    if ((type(vector[i]) != int) and (type(vector[i]) != float) and (type(vector[i]) != complex)):
      inputStatus = False
      print("Invalid Input")
  # If the input is valid the function continues to compute the 2-norm
  if inputStatus == True:
    result = 0
# This for loop will compute the sum of the squares of the elements of the vector. 
    for i in range(len(vector)):
      result = result + (vector[i]**2)
    result = result**(1/2)
    return result

def dot(vector01,vector02):
  '''
  This function takes in two vectors as its inputs and produces the resulting dot product. The dot product is computed by multiplying corresponding elements of each vector. Once the multiplication is complete, the numbers are then added together, resulting in an integer.
  '''
  if Check_Vector(vector01) == False:
    print("invalid input")
  if Check_Vector(vector02) == False:
    print("invalid input")
  else:
    n = len(vector01)
    p = len(vector02)
    #The following line is checking that the length of both vectors is equal to ensure that multiplication is possible
    if n==p:
      x=0
      #This for loop is taking each  corresponding element, multiplying them, and adding them together for the final integer.
      for i in range(len(vector01)):
        x += vector01[i]*vector02[i]
      return x 
    else:
      print("invalid input")
      return None

def vecSubtract(vector03,vector04):
  '''
  The function vecSubtract takes in two vectors and returns the difference of the two. After the difference of each pair of corresponding elements has been computed, it then takes the number and puts it in a new list to return the updated vector.
  '''
  if Check_Vector(vector03) == False:
    print("invalid input")
  if Check_Vector(vector04) == False:
    print("invalid input")
  else:
    n = len(vector03)
    p = len(vector04)
    if n==p:
      x=[]
      #this for loop subtracts the two elements and then appends the difference into the new vector, x
      for i in range(len(vector03)):
        x.append(vector03[i]-vector04[i])
      return x
    else:
      print("invalid input")
      return None

def scalarVecMulti(scalar1,vector05):
  '''
  This function takes in a scalar and vector as its inputs. The scalar is multiplied to each element of the vector. The result is then placed into the new vector, x.
  '''
  if Check_Scalar(scalar1) == False:
    print("invalid input")
  if Check_Vector(vector05) == False:
    print("invalid input")
  else:
    x = []
    #This for loop takes the multiplication of each element of the vector and the scalar and appends the product to new vector x.
    for i in range(len(vector05)):
      x.append(vector05[i]*scalar1)
    return x

def normalize(vector07):
  '''
  This function takes a vector and calculates the normalized vector.Then, the normalized vector is calculated by dividing each element in the specified vector by the norm using the two norm.  
  '''
  if Check_Vector(vector07) == False:
    print("Invalid input")
  else:
    norm = twoNorm(vector07)
    x = []
    #This for loop divides each element of vector07 and divides it by the norm of vector07. It then appends the number to the new normalized vector, x.
    for i in range(len(vector07)):
      x.append(vector07[i]/norm)
    return (x)
  
def QR_factor(A):
  '''
  This function computes the Modified Gran-Schmidt of the given matrix. It utilizes the two norm to then normalize the vectors in matrix A, which becomes the Q vector. We then use the dot product function on the normalized vector and the second vector in the matrix. The last function we use is vector subtraction, which calculates each normalized vector in the Q matrix. The final result is the factorized matrix A into matrix Q and an upper triangular matrix R.
  '''
  if Check_Matrix(A) == False:
    print("invalid input")
  else:
    #We have defined these lengths as variables  to use throughout the function
    m=len(A[0])
    n=len(A)
    V=A
    R=[[0]*n for i in range(n)]
    Q=[[0]*m for i in range(n)]
    for i in range(n):
      #This for loop calculates the norm of the matrix and normalize the first vector
      R[i][i]=twoNorm(V[i])
      Q[i]=normalize(V[i])
      #This for loop normalizes the rest of the vectors in the matrix by the use of the dot product function, scalar multiplication, and vector subtraction.
      for j in range(i+1,n):
        R[j][i]=dot(Q[i],V[j])
        temp=scalarVecMulti(R[j][i],Q[i])
        V[j]=vecSubtract(V[j],temp)
    return [Q,R]

QR=QR_factor(A)
Q=QR[0]
R=QR[1]

def Trans_matvecMulti(Q,y):
  '''
  This function calculates the product of the transpose of Q, which is a matrix, and y, the given data points using matrix, vector multiplication. Instead of calculating the transpose of Q, we can just use Q because this function took a list of rows, not columns, so this allows us to calculate the multiplication because Q is a list of columns.
  '''
  new_matrix=[]
  for i in range(len(Q)):
    mult = 0
    for j in range(len(y)):
      #This calculates each element in matrix Q times the respected element in vector y.
      mult +=Q[i][j]*y[j]
      #This line appends each value froom the above for loop function to the new matrix.
    new_matrix.append(mult)
  return new_matrix

#here we assign the previous function to a value to use in the backsub function.
b=Trans_matvecMulti(Q,y)

def back_sub(R,b):
  '''
  This function takes all of the previously calculated vectors and matrices, Q,R, the data points y, and R and solves for c. The variable c represents the coefficients of the polynomial function from the given the data points in regards to the x values.
  '''
  r=len(b)-1
  c=[0]*len(b)
  c[r]=b[r]/R[r][r]
  #this for loop takes len b in reversed order to isolate c.
  for i in reversed(range(len(b))):
    c[i]=b[i]
    #This for loop calculates each coefficient in vector c.
    for j in range(len(i+1,len(b))):
      c[i]=c[i]-c[j]*R[j][i]
      c[i]=c[i]/R[i][i]
  return c

print(A)
print(QR_factor(A))
print(back_sub(R,b))
