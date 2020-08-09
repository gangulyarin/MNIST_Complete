from sklearn import tree

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini")
}