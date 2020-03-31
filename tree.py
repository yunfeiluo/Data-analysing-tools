import csv
import math

class DataSet:
    """
    This class reads the dataset from a csv file, given the file path as a string.
    It exposes the following class members:

        attributes: a list of strings representing the name of each attribute
        domains: a list of lists indicating the possible values each attribute
                 in self.attributes can take in the provided data
        examples: a list of lists, with each element representing a datapoint
    """
    def __init__(self, path_to_csv):
        with open(path_to_csv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            self.attributes = next(csvreader)
            self.examples = [row for row in csvreader]
            self.domains = [list(set(x)) for x in zip(*self.examples)]

    def set_attrs(self, attrs):
        self.attributes = attrs

    def set_examples(self, exs):
        self.examples = exs

    def set_domains(self, doms):
        self.domains = doms

#For missing data
def ignore(dataset, feature):
    yes = 0
    no = 0
    for e in dataset.examples:
        if e[feature] == 'Yea' or e[feature] == 'Aye':
            yes += 1
        elif e[feature] == 'Nay' or e[feature] == 'No':
            no += 1
    if yes > no:
        return 'Yea'
    else:
        return 'Nay'

def getValue(dataset, value, feature):
    #'''
    if value == 'Yea' or value == 'Aye':
        return 'Yea'
    elif value == 'Nay' or value == 'No':
        return 'Nay'
    elif value == 'Republican' or value == 'Democrat':
        return value
    else:
        return ignore(dataset, feature)
    #'''
    #return value

class Node:
    """
    This class represents an internal node of a decision tree.
    `test_attr` is the index of the attribute to test at this node.
    `test_name` is the human-readable name of that attribute.
    The Node stores a dictionary `self.children` that maps values of the test
    attribute to subtrees, where each subtree is either a Node or a Leaf.
    """
    def __init__(self, test_attr, test_name=None):
        self.test_attr = test_attr
        self.test_name = test_name or test_attr
        self.children = {}

    def classify(self, example):
        """Classify an example based on its test attribute value."""

        # TODO: Implement the classify function here and in the Leaf class
        return self.children[getValue(data, example[self.test_attr], self.test_attr)].classify(example)

    def add_child(self, val, subtree):
        """Add a child node, which could be either a Node or a Leaf."""
        self.children[val] = subtree

    def show(self, level=1):
        """Print a human-readable representation of the tree"""
        print('Test:', self.test_name)
        for (val, subtree) in self.children.items():
            print(' ' * 4 * level, "if", self.test_name, '=', val, '==>', end=' ')
            if isinstance(subtree, Leaf):
                subtree.show()
            else:
                subtree.show(level + 1)


class Leaf:
    """A Leaf holds only a predicted class, with no test."""
    def __init__(self, pred_class):
        self.pred_class = pred_class

    def classify(self, example):
        # TODO: Implement the classify function here
        return self.pred_class

    def show(self):
        """This will be called by the Node `show` function"""
        print('Predicted class:', self.pred_class)


def learn_decision_tree(dataset, target_name, feature_names, depth_limit):
    """
    Trains a decision tree on the provided dataset.
    The `target_name` parameter is the name of the attribute to be predicted.
    The `feature_names` are the names of input attributes that should be used to split the data.
    Finally, `depth_limit` is a parameter to control overfitting by cutting off the tree after
    a certain depth and predicting the plurality class at that split.

    This function should return a decision tree learned from the data.
    """
    domains = dataset.domains
    target = dataset.attributes.index(target_name)
    features = [dataset.attributes.index(name) for name in feature_names]

    def decision_tree_learning(examples, attrs, parent_examples=(), depth=0):
        """
        This function signature is written to match the pseudocode
        on p. 702 of Russell and Norvig. We recommend following that
        pseudocode to implement your decision tree.
        Note that we are adding an argument for the current depth, so you can
        keep track of the depth limit.

        This function should return the decision tree that has been learned.
        """

        # TODO: Implement the logic for learning the decision tree
        # You must also implement the entropy and information gain functions below.
        # We recommend adding your own helper functions below too, but don't remove
        # any of the provided code.
        if len(examples) == 0:
            return pluralityValue(target, parent_examples)
        elif allSame(examples, target):
            return Leaf(examples[0][target])
        elif len(attrs) == 0 or depth >= depth_limit:
            return pluralityValue(target, examples)
        A, v = maximize(features, examples)
        tree = Node(A)
        attrs.remove(A)
        for vk in v:
            exs = vk
            subtree = decision_tree_learning(v[vk], attrs, examples, depth + 1)
            tree.add_child(vk, subtree)
        return tree

    #helper functions:
    def allSame(examples, target):
        judg = getValue(dataset, examples[0][target], target)
        for e in examples:
            if getValue(dataset, e[target], target) != judg:
                return False
        return True

    #PLURALITY-VALUE
    def pluralityValue(target, examples):
        outputs = {}
        for e in examples:
            value = getValue(dataset, e[target], target)
            try:
                outputs[value] += 1
            except:
                outputs[value] = 1
        common = None
        for output in outputs:
            if common == None:
                common = output
            else:
                if outputs[output] > outputs[common]:
                    common = output
                elif outputs[output] == outputs[common]:
                    curr = [output, common]
                    curr.sort()
                    common = curr[0]
        return Leaf(common)

    #IMPORTANCE
    def maximize(features, examples):
        resg = None
        resf = None
        resset = None
        for feature in features:
            cs = {}
            for e in examples:
                value = getValue(dataset, e[feature], feature)
                try:
                    cs[value].append(e)
                except:
                    cs[value] = [e]
            children = []
            for c in cs:
                children.append(cs[c])
            gain = information_gain(examples, children)
            if resf == None:
                resf = feature
                resg = gain
                resset = cs
            else:
                if gain > resg:
                    resg = gain
                    resf = feature
                    resset = cs
                elif gain == resg:
                    if feature < resf:
                        resf = feature
                        resset = cs
        return resf, resset

    def entropy(examples):
        """Takes a list of examples and returns their entropy w.r.t. the target attribute"""

        # TODO: Implement the entropy function
        res = 0
        le = len(examples)
        properties = {}
        for e in examples:
            curr = None
            try:
                curr = properties[e[target]]
            except:
                properties[e[target]] = 1
                continue
            properties[e[target]] += 1
        for p in properties:
            pi = (properties[p] / le)
            res += (pi * math.log(pi, 2))
        return -1 * res

    def information_gain(parent, children):
        """
        Takes a `parent` set and a subset `children` of the parent.
        Returns the information gain due to splitting `children` from `parent`.
        """

        # TODO: Implement the information gain
        pc = 0
        lp = len(parent)
        for child in children:
            pc += ((len(child) / lp) * entropy(child))
        return entropy(parent) - pc

    return decision_tree_learning(dataset.examples, features)

if __name__ == '__main__':
    """
    You can use this area to test your implementation and to generate
    output for the assignment. The autograder will ignore this area.
    """

    ############################
    ###### Example usage: ######
    ############################

    data = DataSet("./congress_data.csv")

    # An example of learning a decision tree to predict party affiliation
    # based on the values of votes 4-7

    choices = ["Party", "Vote4", "Vote5", "Vote6", "Vote7", "Vote200", "Vote300", "Vote112", "Vote118"]

    target = choices[0]
    attrs = [choices[2], "Vote4", "Vote6", "Vote7", "Vote200", "Vote300", "Vote112", "Vote118"]

    t = learn_decision_tree(data, target, attrs, 3)
    t.show()

    correct = 0
    total = 0
    target = data.attributes.index(target)
    for e in data.examples:
        total += 1
        pred = t.classify(e)
        if pred == e[target]:
            correct += 1
    print("correct prediction: " + str(correct / total))