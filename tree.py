import csv
import math
import sys

class DataSet:
    """
    This class reads the dataset from a csv file, given the file path as a string.
    It exposes the following class members:

    @attribute attributes: list of features name (string)
    @attribute domains: list of lists indicating the possible values each attribute in self.attributes can take in the provided data
    @attribute examples: list of data points
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

class Node:
    """
    This class represents an internal node of a decision tree.
    @attribute test_attr: the index of the features to test at this node
    @attribute test_name: the name (string) of the test feature
    @attribute children: dictionary, map: value_of_test_feature -> subtrees
    """
    def __init__(self, test_attr, test_name=None):
        self.test_attr = test_attr
        self.test_name = test_name or test_attr
        self.children = {}

    def classify(self, example):
        """Classify an example based on its test attribute value."""
        return self.children[getValue(data, example[self.test_attr], self.test_attr)].classify(example)

    def add_child(self, val, subtree):
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
        return self.pred_class

    def show(self):
        """This will be called by the Node `show` function"""
        print('Predicted class:', self.pred_class)


def learn_decision_tree(dataset, target_name, feature_names, depth_limit):
    """
    @param dataset: dataset
    @param target_name: name of target feature
    @depth_limit: max depth the tree can have
    """
    domains = dataset.domains
    target = dataset.attributes.index(target_name)
    features = [dataset.attributes.index(name) for name in feature_names]

    def decision_tree_learning(examples, attrs, parent_examples=(), depth=0):
        """
        @param examples: list of data points
        @param atts: list of features
        @return the learned tree
        """
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

    #PLURALITY-VALUE (helper function)
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

    #IMPORTANCE (helper function)
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
        """
        @param examples: list of data points
        @return the entropy of this set w.r.t target feature
        """
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
        @param parent: set of data points before splitting
        @param children: one subset of data points splitted from parent set
        @return information gain due to splitting the parent into children
        """
        pc = 0
        lp = len(parent)
        for child in children:
            pc += ((len(child) / lp) * entropy(child))
        return entropy(parent) - pc

    return decision_tree_learning(dataset.examples, features)

if __name__ == '__main__':
    # Get required parameters from command line arguments
    data = DataSet(sys[1])
    choices = sys[2] # list of features, the 1st entry is the target feature
    depth = None
    try:
        depth = sys[3]
    except:
        depth = 3 # default

    target = choices[0]
    attrs = choices[1:]

    t = learn_decision_tree(data, target, attrs, depth)
    t.show()
    
    # Calculate Accuracy
    correct = 0
    total = 0
    target = data.attributes.index(target)
    for e in data.examples:
        total += 1
        pred = t.classify(e)
        if pred == e[target]:
            correct += 1
    print("correct prediction: " + str(correct / total))
