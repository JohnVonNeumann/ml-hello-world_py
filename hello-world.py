from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]] # features of 1 = bumpy texture, feaures of 0 = smooth texture
# label 0 = apples, label 1 = oranges
labels = [0, 0, 1, 1]
# use numbers to ensure the data is crunchable by the program
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# -------------------------------


print(clf.predict([[160, 0]])) # our input/question wth params

# output will be 0 if apple and 1 if orange
