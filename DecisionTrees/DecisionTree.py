import pandas as pd
import numpy as np
import operator

#Authors: Kavya Sethuram, Rasika Guru, Roshani Mangalore
# Class to store node
class tree_node:
    def __init__(self, label=None, attr=None):
        self.label = label
        self.attr = attr
        self.branches = {}

#function to print the decision tree
def print_tree(node, level):
    print('\t' * level + node.attr)
    for branch in node.branches:
        if node.branches[branch].attr != None:
            print('\t' * (level + 1) + branch)
            print_tree(node.branches[branch], level + 1)
        else:
            print('\t' * (level + 1) + branch + '-' + node.branches[branch].label)

# Function to break ties between attributes when having same gain
# Attribute occuring at the earlier in the attribute list given preference
def break_ties(attributes, attr_order):
    if len(attributes) == 1:
        return (attributes[0])
    for attr in attr_order:
        if attr in attributes:
            return (attr)

# Calculating most common labels in a list of labels
# Used to find majority class in remaining data after we run out of attributes
def calc_most_common(l):
    count_dict = {}
    for ele in l:
        if ele not in count_dict:
            count_dict[ele] = 1
        else:
            count_dict[ele] += 1
    max_count = 0
    for key in count_dict:
        if count_dict[key] > max_count:
            max_count = count_dict[key]
            most_common = key
    c = 0
    for key in count_dict:
        if count_dict[key] == max_count:
            c += 1
    if c > 1:
        most_common = 'Tie'
    return (most_common)

# Calculate entropy of a given set
def calculate_entropy(Y, label_name):
    classes = Y[label_name].unique()
    class_dict = {c: 0 for c in classes}
    Y = np.asarray(Y[label_name])
    for y in Y:
        class_dict[y] += 1
    tot = len(Y) * 1.0
    for key in class_dict:
        class_dict[key] /= tot
    entropy = 0
    for key in class_dict:
        entropy += -1 * class_dict[key] * np.log2(class_dict[key])
    return (entropy)

# Calculate information gain for a given set and attribute
def calculate_gain(S, A, label_name):
    Ent_S = calculate_entropy(S[[label_name]], label_name)
    values_A = S[A].unique()
    avg_ent_AVs = 0
    val_dict = {v: 0 for v in values_A}
    A_data = np.asarray(S[A])
    for aval in A_data:
        val_dict[aval] += 1
    tot = len(A_data) * 1.0
    for key in val_dict:
        val_dict[key] /= tot
        val_dict[key] *= calculate_entropy(S.loc[S[A] == key], label_name)
        avg_ent_AVs += val_dict[key]
    return (Ent_S - avg_ent_AVs)

# Build the decision tree
def decision_tree(df, label_name, remaining_attributes, possible_values_dict, attr_order):
    node = tree_node()  # Create New Node
    if len(df[label_name].unique()) == 1:  # Recursion end condition 1 - pure class
        node.label = df[label_name].unique()[0]
    elif len(remaining_attributes) == 0:  # Recursion end condition 2 - no more attributes
        labels = df[label_name]
        node.label = calc_most_common(labels)
    else:
        max_gain = 0.0
        gain_dict = {}
        for attr in remaining_attributes:  # Loop through all attributes
            gain = calculate_gain(df, attr, label_name)  # and find the one with highest gain
            gain_dict[attr] = gain
            if gain > max_gain:
                max_gain = gain
        max_gain_attributes = []
        for attr in gain_dict:
            if gain_dict[attr] == max_gain:
                max_gain_attributes.append(attr)
        max_gain_attr = break_ties(max_gain_attributes, attr_order)
        node.attr = max_gain_attr
        A_values = possible_values_dict[max_gain_attr]
        for value in A_values:  # Branch based on attr with max gain
            df_subset = df.loc[df[max_gain_attr] == value]
            if df_subset[
                label_name].count() == 0:  # If the subset with particular value of attribute has no record, add leaf node
                node.branches[value] = tree_node(label='Tie')
            else:  # Else rucurse and add the returned node to the value branch
                node.branches[value] = decision_tree(df_subset, label_name, remaining_attributes - set([max_gain_attr]),
                                                     possible_values_dict, attr_order)
    return (node)

#function to predict the class label for Test Data
def predict(node, predict_data):
    if node.label != None:
        return node.label
    else:
        return predict(node.branches[predict_data[node.attr]], predict_data)

if __name__ == '__main__':
    filename = 'dt-data.txt'
    label_name = 'Enjoy'
    columns = []
    with open(filename, 'r') as in_file:
        data_list = []
        for line in in_file:
            if line[0] == '(':
                columns = line[1:-2].split(',')
                columns = [col.strip() for col in columns]
            elif line[0] == '\n':
                continue
            else:
                line = line[4:-2].split(',')
                line = [l.strip() for l in line]
                temp_dict = {}
                for i in range(0, len(columns)):
                    temp_dict[columns[i]] = line[i]
                data_list.append(temp_dict)
        df = pd.DataFrame(data_list)
    attr_order = columns
    attributes = set(columns) - set([label_name])
    possible_values_dict = {}
    for attr in attributes:
        possible_values_dict[attr] = df[attr].unique()
    root = decision_tree(df, label_name, attributes, possible_values_dict, attr_order)
    print_tree(root, 0)
    predict_data = {'Occupied': 'Moderate', 'Price': 'Cheap', 'Music': 'Loud', 'Location': 'City-Center', 'VIP': 'No',
                    'FavouriteBeer': 'No'}
    prediction = predict(root, predict_data)
    print("Prediction:", prediction)