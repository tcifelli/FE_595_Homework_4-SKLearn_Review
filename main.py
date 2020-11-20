import pandas as pd
from sklearn.linear_model import LinearRegression #used for Boston data
from sklearn.cluster import KMeans #used with Iris and Wine data
from sklearn.datasets import load_boston, load_iris, load_wine #data loaders
import matplotlib.pyplot as plt #plotting package

#run a linear regression on the Boston dataset and rank the input variables
#by the magnitude of their coefficients
def bostonRunner(printResults = True):
    names = load_boston()['feature_names'] #the variable names
    X, y = load_boston(return_X_y=True) #get our training data
    model = LinearRegression() #create the LinearRegression object
    model.fit(X, y) #fit the model to the data
    coeffs = pd.DataFrame([(name, coeff) for name, coeff in zip(names, model.coef_)],
                          columns=["Parameter Name", "Coefficient"]) #pair the names and coeffs
    coeffs.sort_values("Coefficient", ascending=False, inplace=True) #sort by coefficient

    #print rankings to the console
    if printResults:
        print("Boston Dataset Results:")
        print(coeffs)

    return coeffs


#generate elbow plots for KMeans on the Iris and Wine datasets
def elbowAnalyzer():
    irisX, irisy = load_iris(return_X_y=True) #load in the datasets
    wineX, winey = load_wine(return_X_y=True)

    #loop for different values of K and log the sum of squared errors (inertias) of the models
    irisErrors = []
    wineErrors = []
    for i in range(1, 11):
        irisModel = KMeans(n_clusters=i).fit(irisX, irisy) #fit both models
        wineModel = KMeans(n_clusters=i).fit(wineX, winey)

        irisErrors.append(irisModel.inertia_) #extract their SSEs
        wineErrors.append(wineModel.inertia_)

    #generate the elbow plot for the Iris data
    plt.figure()
    plt.plot(range(1, 11), irisErrors)
    plt.title("Sum of Squared Errors for Iris Dataset")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.savefig("Output Graphs/IrisElbowPlot.png")
    plt.show()

    #generate the elbow plot for the Wine data
    plt.figure()
    plt.plot(range(1, 11), wineErrors)
    plt.title("Sum of Squared Errors for Wine Dataset")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.savefig("Output Graphs/WineElbowPlot.png")
    plt.show()

    #for both of these datasets we see that the elbow plot "kinks" around the value of three, implying that using fewer
    #clusters dramatically reduces the quality of our model and that using more suffers from diminishing returns


def main():
    bostonRunner() #run first part of the assignment
    elbowAnalyzer() #run the second part

if __name__ == "__main__":
    main()
