from mpl_toolkits.mplot3d import Axes3D
import time, datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import math, sys, os







def calcEDistance(data, i, j):
   pi = data[i]['Coordinate']
   pj = data[j]['Coordinate']
   pow((pi[0] - pj[0]),2)
   dist = np.sqrt(pow((pi[0] - pj[0]),2)+pow((pi[1] - pj[1]),2)+pow((pi[2] - pj[2]),2))
   return dist



def distFromPtToTri(point, triangle):
    """
    This function is used to calculate the distance from a point to a triangle in 3D space

    input: point    [1D array]  the x, y, z coordinates of the point
           triangle [2D array]  the x, y, z coordinates of 3 vertices of a triangle

    output: dist  [scalar]    the shortest distance from the point to the plane that contains the triangle
            ptp   [1D array]  coordinate of the projected point within the plane


    method is based on vector operation. assume the given point is P and the three vertices of the triangle are A, B, C
    vectorAB:   vector start from A to B
    vectorAC:   vector start from A to C
    vectorAP:   vector start from A to P
    triNorm:    unit normal vector to the plane of ABC
    signedDist: the vertical (shortest) distance from P to ABC
                it could be positive or negative, but the sign only indicates whether the vector AP is inward or outward
                the plane ABC. We actually do not care about this sign.
    dis:        the absolute value of the signedDist
    ptp:        project the input point to the plane ABC. ptp is the coordinate of the projected point

    """
    pta = triangle[0]
    ptb = triangle[1]
    ptc = triangle[2]
    vectorAB = np.subtract(ptb, pta)
    vectorAC = np.subtract(ptc, pta)
    vectorAP = np.subtract(point, pta)
    triNorm = np.cross(vectorAB, vectorAC)
    triNorm = np.divide(triNorm, np.linalg.norm(triNorm, ord=2))
    signedDist = np.dot(vectorAP, triNorm)
    dist = abs(signedDist)
    projVector = vectorAP - signedDist * triNorm
    ptp = pta + projVector
    return dist, ptp


def ptInTriangle(projPoint, triangle, tol):
    """
    This function is used to check whether the projected point is within the triangle ABC

    input: projPoint   [1D array]  the x, y, z coordinates of the projected point from last function
           triangle    [2D array]  the x, y, z coordinates of 3 vertices of a triangle used in last function

    output:            [boolean]    whether the projected point is in the triangle
    """
    pta = triangle[0]
    ptb = triangle[1]
    ptc = triangle[2]
    vectorAB = np.subtract(ptb, pta)
    vectorAC = np.subtract(ptc, pta)
    vectorBC = np.subtract(ptc, ptb)
    vectorAP = np.subtract(projPoint, pta)
    vectorBP = np.subtract(projPoint, ptb)
    areaABC = 0.5 * np.linalg.norm(np.cross(vectorAB, vectorAC), ord=2)
    areaABP = 0.5 * np.linalg.norm(np.cross(vectorAB, vectorAP), ord=2)
    areaACP = 0.5 * np.linalg.norm(np.cross(vectorAC, vectorAP), ord=2)
    areaBCP = 0.5 * np.linalg.norm(np.cross(vectorBC, vectorBP), ord=2)
    if abs(areaABP + areaACP + areaBCP - areaABC) < tol:
        return True
    else:
        return False



def readData(dataPath):

    input = pd.read_csv(dataPath, header=None)
    data = {}
    indexList = []
    for index in range(input.shape[0]):
        x = input.iloc[index, 0]
        y = input.iloc[index, 1]
        z = input.iloc[index, 2]
        data[index] = {}
        data[index]['Coordinate'] = []
        data[index]['Coordinate'].append(x)
        data[index]['Coordinate'].append(y)
        data[index]['Coordinate'].append(z)
        data[index]['BB'] = {}
        data[index]['BB']['X'] = {}
        data[index]['BB']['X']['min'] = x - 0.5
        data[index]['BB']['X']['max'] = x + 0.5
        data[index]['BB']['Y'] = {}
        data[index]['BB']['Y']['min'] = y - 0.5
        data[index]['BB']['Y']['max'] = y + 0.5
        data[index]['BB']['Z'] = {}
        data[index]['BB']['Z']['min'] = z - 0.5
        data[index]['BB']['Z']['max'] = z + 0.5
        data[index]['GridLabel'] = {}
        data[index]['GridLabel']['X'] = np.floor(x)
        data[index]['GridLabel']['Y'] = np.floor(y)
        data[index]['GridLabel']['Z'] = np.floor(z)
        indexList.append(index)
    return data

def buildAdjacencyDict_grid(data):

    indexList = list(data.keys())
    # GRID algorithm:
    adjacency_dict_GRID = {}
    for i in indexList:
        adjacency_dict_GRID[i] = []
        for j in range(0, i):
            if abs(data[i]['GridLabel']['X'] - data[j]['GridLabel']['X']) > (1 + epsilon):
                continue
            if abs(data[i]['GridLabel']['Y'] - data[j]['GridLabel']['Y']) > (1 + epsilon):
                continue
            if abs(data[i]['GridLabel']['Z'] - data[j]['GridLabel']['Z']) > (1 + epsilon):
                continue
            if calcEDistance(data, i, j) < threshold + epsilon:
                adjacency_dict_GRID[i].append(j)
                adjacency_dict_GRID[j].append(i)
    return adjacency_dict_GRID


def saveAdjacencyFile(resultPath, adjacency_dict_GRID, filename):
    """From dictionary to csv"""

    adjacencyPath = resultPath + filename+'.csv'
    adjacency_df = pd.DataFrame.from_dict(adjacency_dict_GRID, orient='index', columns=None,dtype=int)
    adjacency_df.to_csv(adjacencyPath)



def createDecimatedDict(data, adjacency_dict_GRID, N_adj_threshold):
    """Finds the boundary points among all the points in the input file using the adjacency file and the threshold

    Inputs:
    data is the input file,
    adjacency_dict_GRID is the dictionary of adjacency, including adjacency information and
    the N_adj_threshold is the number of points above which if a point has adjacent points is called interior

    Outputs:
    Refined data file, with adding information about the type of points, and
    boundaryPointsDict which is a dictionary containing the original indices of points that are boundary based on the
    criteria as the keys and the coordinates of the boundary points as the values.

    Note: We assume there is no exterior point
    """
    boundaryPointsCounter = 0
    boundaryPointsDict = {}
    indexList = list(data.keys())
    for i in indexList:
        data[i]['E_adj']= adjacency_dict_GRID[i]
        if len(data[i]['E_adj']) >= N_adj_threshold:
            data[i]['E_type'] = 'Int'
        # elif len(data[i]['E_adj']) < N_adj_threshold:
        #     data[i]['E_type'] = 'Ext'
        else:
            data[i]['E_type'] = 'Boundary'
            boundaryPointsDict[i] = data[i]['Coordinate']
            boundaryPointsCounter+=1
    ratio =  boundaryPointsCounter/len(indexList)
    return data, boundaryPointsDict, ratio


def saveDecimatedFile(resultPath, boundaryPointsDict, filename):
    """From Decimated points dictionary to csv"""
    decimatedPath = resultPath + filename+'.csv'
    decimated_df = pd.DataFrame.from_dict(boundaryPointsDict, orient='index', columns=['X', 'Y', 'Z'])
    decimated_df.to_csv(decimatedPath)


def saveTrianglesFile(resultPath, triangleDict, filename):
    """From Decimated points dictionary to csv"""
    TriPath = resultPath + filename+'.csv'
    triangles_df = pd.DataFrame.from_dict(triangleDict, orient='index', columns=['Vertice1', 'Vertice2', 'Vertice3'])
    triangles_df.to_csv(TriPath)


def plotBoundaryInteriorPoints(data, resultPath, filename, N_adj_threshold):

    x_int_list, y_int_list, z_int_list = [], [], []
    x_Boundary_list, y_Boundary_list, z_Boundary_list = [], [], []
    indexList = list(data.keys())
    for i in indexList:
        if data[i]['E_type'] == 'Int':
            x_int_list.append(data[i]['Coordinate'][0])
            y_int_list.append(data[i]['Coordinate'][1])
            z_int_list.append(data[i]['Coordinate'][2])

        else:
            x_Boundary_list.append(data[i]['Coordinate'][0])
            y_Boundary_list.append(data[i]['Coordinate'][1])
            z_Boundary_list.append(data[i]['Coordinate'][2])
    caption = 'Number of interior points: '+ str(len(x_int_list))+ '\n' +'Number of boundary points: '+ str(len(x_Boundary_list)+ '\nRatio :'+str(len(z_Boundary_list)/len(z_int_list)))
    plotname = 'Boundary_and_interior_'+filename
    fig = plt.figure(figsize = [9, 7])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_int_list, y_int_list, z_int_list, 'b',  marker='o', label='Interior')
    # ax.scatter(x_Ext_list, y_Ext_list, z_Ext_list, 'g' , marker='^', label='Exterior')
    ax.scatter(x_Boundary_list, y_Boundary_list, z_Boundary_list, 'r', marker='x', label='Boundary')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    fig.text(.5, .05, caption, ha='center')
    plt.title('The boundary and interior points according to '+str(N_adj_threshold) +' adjacent number of points')
    plt.legend(loc='lower right')
    fig.savefig(resultPath + plotname + ".pdf")


def createConvexHull_Qhull(data, savefig = True):
    indexList = list(data.keys())

    # points Coordinates
    pts = np.zeros([len(indexList), 3])
    for i in indexList:
        pts[i, :] = data[i]['Coordinate']

    hull = ConvexHull(pts)
    convexHullpts = np.zeros([len(list(hull.vertices)), 3])
    index = 0
    for v in list(hull.vertices):
        convexHullpts[index, :] = data[v]['Coordinate']
        index += 1
    if savefig == True:
        plotname = 'ConvexHull'
        plotConvexHullFacesAndPoints(resultPath, hull, pts, plotname)

    return hull, convexHullpts


def plotConvexHullFacesAndPoints(resultPath, hull, pts, plotname):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")


    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))
    plt.show()
    fig.savefig(resultPath + plotname + ".pdf")

def createTriangles(data, hull, resultPath, visulization = True):
    triangleDict = {}
    vertices_points = np.zeros([hull.vertices.shape[0], 3])
    for tri in range(hull.simplices.shape[0]):
        triangleDict[tri] = list(hull.simplices[tri, :])

    vertindex = 0
    for vert in hull.vertices:
        vertices_points[vertindex, :] = data[vert]['Coordinate']
        vertindex+=1

    indexList = list(data.keys())

    # points Coordinates
    pts = np.zeros([len(indexList), 3])
    for i in indexList:
        pts[i, :] = data[i]['Coordinate']

    if visulization:
        plotname = 'HullVertices_Triangles_Boundary'
        x_Boundary_list, y_Boundary_list, z_Boundary_list = [], [], []
        indexList = list(data.keys())
        for i in indexList:
            if data[i]['E_type'] == 'Boundary':
                x_Boundary_list.append(data[i]['Coordinate'][0])
                y_Boundary_list.append(data[i]['Coordinate'][1])
                z_Boundary_list.append(data[i]['Coordinate'][2])

        fig2 = plt.figure(figsize=[9, 7])
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(vertices_points[:, 0], vertices_points[:, 1], vertices_points[:, 2], 'b', marker='o',
                   label='Convex Hull Vertices')
        ax.scatter(x_Boundary_list, y_Boundary_list, z_Boundary_list, 'r', marker='x', label='Boundary Points')

        triIndex = 0
        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 'b', marker='o', label='Points')
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            if triIndex == 0:
                ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-", label='Triangles')
            else:
                ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
            triIndex+=1

        plt.legend()
        fig2.savefig(resultPath + plotname + ".pdf")
    filename = 'Triangles'
    saveTrianglesFile(resultPath, triangleDict, filename)
    return triangleDict





def isConvex(data, boundaryPointsDict, triangleDict):

    """This function determines whether the point cloud is a convex or not based on the deistance of the boundary points
    to the triangles of the convex hull, The algorithm is as the following:
    For each point we check whether there exists at least one triangle to which the distance is zero:
    If such triangle is not found even for one point, the point cloud (boundary points) is non convex.

    Input:
            data (coordinate of the points)
            boundaryPointsDict: boundary points coordinates
            triangleDict: triangles including the indices of the corner in the reference data

    data:
            Boolean: Convex or not

    """

    # This loop the boundary points:
    for bdrPntIdx in list(boundaryPointsDict.keys()):
        point = np.array(data[bdrPntIdx]['Coordinate'])

        #The flag showing whether a point is on at least of the triangles after looping all the triangles:
        PntonConvexFlag = False
        for tri in list(triangleDict.keys()):
            triangle = np.zeros([3,3])
            for corner in range(0,3):
                triangle[corner, :] = data[triangleDict[tri][corner]]['Coordinate']
            dis, ptp = distFromPtToTri(point, triangle)
            isIn = ptInTriangle(ptp, triangle, approximation_treshold)

            # if we find a triangle for the selected point such that their distance is zero, we dont need to check the
            # distance of that particular point to the rest of triangles, so we continue by selecting the next point
            if dis == 0:
                onConvexFlag = True
                continue

        # If at the end of the loop still the flag is Flag is false means that the particular point is not on none of the
        # triangles, so we can immediately decide the shape is non convex, and there is no need to check other points
        if not PntonConvexFlag:
            return False

    # at the end of checking all the points, if there is no false as return we conclude that all the points are on the
    # convex hall and the shape is convex

    return True


# parameters
dataPath = "/Users/marjanhosseini/Documents/Courses/Algorithm/HW2/CSE5500-HW2/shape4-coord copy.csv"
#dataPath = "/Users/marjanhosseini/Documents/Courses/Algorithm/HW2/CSE5500-HW2/Test.csv"
resultPath = "/Users/marjanhosseini/Documents/Courses/Algorithm/HW2/Results/"
adjacencyFilePath = "/Users/marjanhosseini/Documents/Courses/Algorithm/HW2/Results/adjacency.csv"
threshold = 1
epsilon = pow(10,-6)
n_runs = 1
N_adj_threshold = 6
approximation_treshold = pow(10, -3)




files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(dataPath):
    for file in f:
        if '.csv' in file:
            files.append(file)
            #files.append(os.path.join(r, file))

filename, extension = os.path.splitext(files[0])


# read data points from the path
data = readData(dataPath)

# create adjacency dictionary
adjacencyDict = buildAdjacencyDict_grid(data)

# save the adjacency dictionary to the result path
saveAdjacencyFile(resultPath, adjacencyDict, 'adjacency_4')

# Create decimated dictionary (the points having less than 6 points as neighbors are boundary points and the ones
# having equal or above 6 neighbors are interior points (according to the instruction we assume we do not have any
# exterior points)
data, boundaryPointsDict, ratio = createDecimatedDict(data, adjacencyDict, N_adj_threshold)

print('The ratio of the decimated file size to the original file size: ', ratio)

# save the decimated points in the result path
saveDecimatedFile(resultPath, boundaryPointsDict, 'Decimated_points_4')

# plot interior and boundary points and the triangles
plotBoundaryInteriorPoints(data, resultPath, N_adj_threshold)

# find the convex hull using scipy library and returns the hull object and the points of the convex hull
# and plot convex hull and the faces of the hull
# source: scipy.spatial.ConvexHull library
# documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
# This function plots all the points and the triangles of the convex hull
hull, convexHullpts = createConvexHull_Qhull(data, savefig = False)

# Create, plots and saves triangles of the convex hull which is obtained from previous step
triangleDict = createTriangles(data, hull, resultPath, visulization = True)


convexity = isConvex(data, boundaryPointsDict, triangleDict)