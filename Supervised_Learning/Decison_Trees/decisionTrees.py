import numpy as np

class DecisionTrees:

    def __init__(self, data, p1root=None):
        self.data = data
        if (p1root):
            self.p1root = p1root
        else:
            self.p1root = self.split(data)

    def entropy(self, p):
        if p==0 or p==1:
            return 0
        return -1 * p * np.log2(p) - 1 * (1-p) * np.log2(1-p)
    
    def split(self, arr):
        if arr.shape[0] == 0:
            return 0  # Or handle as appropriate for your tree logic (e.g., return 0.5 for max uncertainty)
        return np.count_nonzero(arr[:,-1]==1)/arr.shape[0]
    
    def informationGain(self, feature):
        if self.data.shape[0] == 0:
            return 0 # No data, so no information gain

        g1 = self.data[self.data[:, feature] == 1]
        # print(g1)
        g2 = self.data[self.data[:, feature] != 1]      
        # print(g2) 

        p1left = self.split(g1)
        # print(p1left)
        p1right = self.split(g2)
        # print(p1right)

        wleft = g1.shape[0]/self.data.shape[0]
        wright = g2.shape[0]/self.data.shape[0]

        return self.entropy(self.p1root) - (wleft * self.entropy(p1left) + wright * self.entropy(p1right))
    
    def maxInformationGainFeature(self):
        featureInformationGains = np.array([self.informationGain(i) for i in range(self.data.shape[1]-1)])
        chosenFeature = np.argmax(featureInformationGains);
        return chosenFeature
                                           
    
class DecisionTreeNode:

    def __init__(self, data, depth):
        self.data = data
        self.depth = depth
        self.left = None
        self.right = None
        self.label = self.computeLabel()

    def computeLabel(self):
        if self.data.shape[0] == 0:
            return None
        
        return np.argmax([np.count_nonzero(self.data[:,-1]!=1),
                          np.count_nonzero(self.data[:,-1]==1)])
        

    def expand(self):
        DT = DecisionTrees(self.data)
        feature = DT.maxInformationGainFeature()

        dataLeft = self.data[self.data[:, feature] == 1]

        dataRight = self.data[self.data[:, feature] != 1]

        if dataLeft.shape[0]!=0:
            self.left = DecisionTreeNode(dataLeft, self.depth+1)

        if dataRight.shape[0]!=0:
            self.right = DecisionTreeNode(dataRight, self.depth+1)  

def buildTree(Node, maxDepth):

    if Node.depth > maxDepth:
        return
    
    Node.expand()

    if Node.left is not None:
        buildTree(Node.left, maxDepth)
    if Node.right is not None:
        buildTree(Node.right, maxDepth)

if __name__=='__main__':

    data = np.array([[1,1], [1,1], [1,1], [1,1], [1,0],
                     [0,1], [0,0], [0,0], [0,0], [0,0],
                     ])
    
    root = DecisionTreeNode(data,0)
    buildTree(root, 2)
