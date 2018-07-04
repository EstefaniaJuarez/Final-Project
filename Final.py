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
  
def QR_factor(matrix):
  '''
  This function computes the Modified Gran-Schmidt of the given matrix. It utilizes the two norm to then normalize the vectors in matrix A, which becomes the Q vector. We then use the dot product function on the normalized vector and the second vector in the matrix. The last function we use is vector subtraction, which calculates each normalized vector in the Q matrix. The final result is the factorized matrix A into matrix Q and an upper triangular matrix R.
  '''
  if Check_Matrix(matrix) == False:
    print("invalid input")
  else:
    #We have defined these lengths as variables  to use throughout the function
    m=len(matrix[0])
    n=len(matrix)
    V=matrix
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
matrix=[[1,0,1],[2,1,0]]
print(QR_factor(matrix))

QR=QR_factor(matrix)
Q=QR[0]
R=QR[1]

def Trans_matvecMulti(Q,y):
  new_matrix=[]
  for i in range(len(Q)):
    mult = 0
    for j in range(len(y)):
      mult +=Q[i][j]*y[j]
    new_matrix.append(mult)
  return new_matrix

b=Trans_matvecMulti(Q,y)

def back_sub(R,b):
  r=len(b)-1
  c=[0]*len(b)
  c[r]=b[r]/R[r][r]
  for i in reversed(range(len(b))):
    c[i]=b[i]
    for j in range(len(i+1,len(b))):
      c[i]=c[i]-c[j]*R[j][i]
      c[i]=c[i]/R[i][i]
    return c
