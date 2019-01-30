from sklearn import tree

#DataSet
#[height,length,width] in centimeters
X = [[450, 914, 274], [452, 910, 271], [212, 360, 155], [254, 388, 155],[451, 912, 272],
     [121, 152, 5], [121, 164, 4],
     [251, 390, 170], [255, 355, 155], [111, 165, 10], [458, 914, 274]]

Y = ['Bus', 'Bus', 'Car', 'Car', 'Bus', 'Bike', 'Bike', 'Car',
     'Car', 'Bike', 'Bus']

#classifier - DecisionTreeClassifier
clf_tree = tree.DecisionTreeClassifier();
clf_tree = clf_tree.fit(X,Y);

#test_data
test_data = [[452,912,274],[121,155,10],[212,350,155]];

#prediction
prediction_tree = clf_tree.predict(test_data);

print("Prediction of DecisionTreeClassifier:",prediction_tree);
