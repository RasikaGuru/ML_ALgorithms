#Authors: Kavya Sethuram, Rasika Guru, Roshani Mangalore
import pandas as pd
import numpy as np

#Main function
if __name__ == "__main__":
    df = pd.read_csv('pca-data.txt', sep = '\t',header=None)
    data_array = df.as_matrix()
    count = len(data_array)
    mean_list = []

#Computation of Mean for all co-ordinates
    for i in range(3):
        sum = 0
        for j in range(count):
            sum+=data_array[j][i]
        mean_list.append(sum/count)

# Computation of (X minus Mean) matrix
    for i in range(3):
        for j in range(count):
            data_array[j][i]= data_array[j][i]-mean_list[i]

#Initializing Covariance Matrix
    covariance_matrix = np.zeros(shape=(3,3))
#Computation of Covariance Matrix
    covariance_matrix = (np.matrix(data_array).T* np.matrix(data_array))/count

#Computation of Eigen Values and Eigen Vectors from Covariance Matrix
    eigenvalues,eigenvector = np.linalg.eig(covariance_matrix)
    eigen_list=[]
    eigenvector=np.asarray(eigenvector)
    eigenvec = [[eigenvector[0][i],eigenvector[1][i],eigenvector[2][i]] for i in range(3)]

#Select two highest Eigen values and their corresponding Eigen Vectors
    eigen_dict = {eigenvalues[i]:eigenvec[i] for i in range(3)}
    sorted_eigenvalues = sorted(eigen_dict.items(),reverse = True)
    k_dimension_list = []
    tries=0
    for i in range(len(sorted_eigenvalues)):
        if(tries == 2):
            break
        tries = tries + 1
        k_dimension_list.append(sorted_eigenvalues[i][1])

    eigen_matrix = np.array(k_dimension_list)
    data_array_transpose=np.transpose(data_array)

    #Computation of k-dimension matrix
    reduced_dimension_matrix =  np.dot(eigen_matrix,data_array_transpose)
    reduced_dimension_list= [[reduced_dimension_matrix[0,i],reduced_dimension_matrix[1,i]] for i in range(len(reduced_dimension_matrix[0]))]
    reduced_dimension_list=np.matrix(reduced_dimension_list)
    #to print the output on the screen
    print reduced_dimension_list

#to write into a file
with open('outfile.txt',"w") as f:
    for line in reduced_dimension_list:
        np.savetxt(f, line,delimiter='\t',fmt='%.8f')