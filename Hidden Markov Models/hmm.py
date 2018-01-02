def get_neighbours(loc, size):
    x = loc[0]
    y = loc[1]
    neighbours = []
    if x + 1 < size:
        neighbours.append((x + 1, y))
    if y + 1 < size:
        neighbours.append((x, y + 1))
    if x - 1 > 0:
        neighbours.append((x - 1, y))
    if y - 1 > 0:
        neighbours.append((x, y - 1))
    return neighbours


def probable_cells(free, given_dist, dist_to_towers):
    prob = []
    for i in range(len(free)):
        count = 0
        for j in range(len(given_dist)):
            if dist_to_towers[i][j][0] <= given_dist[j] and given_dist[j] <= dist_to_towers[i][j][1]:
                count += 1
        if count == len(given_dist):
            prob.append(free[i])
    return prob


def transition_prob(sdict, neighbours):
    total_transprob = {}
    trans_prob_neighbours = {}
    transition_prob = {}
    for cell in sdict:
        trans_prob_neighbours[cell], transition_prob[cell] = {}, {}
        total_transprob[cell] = 0.0
        allowed = sdict[cell]
        neighbouring_cells = neighbours[cell]
        for step in allowed:
            for neighbour in neighbouring_cells:
                if (neighbour in sdict) and (step + 1 in sdict[neighbour]):
                    if neighbour not in trans_prob_neighbours[cell]:
                        trans_prob_neighbours[cell][neighbour] = 0.0
                    trans_prob_neighbours[cell][neighbour] += 1.0
                    total_transprob[cell] += 1.0
        for neighbour in trans_prob_neighbours[cell]:
            transition_prob[cell][neighbour] = trans_prob_neighbours[cell][neighbour] / total_transprob[cell]
    return transition_prob


def viterbi(free, tower_locations, given_dist, dist_to_towers, possible_states, neighbours, transition_prob):
    step = 0
    states = []
    paths = {}
    paths[step] = {}
    for item in possible_states[step]:
        item = tuple(item)
        paths[step][item] = {}
        paths[step][item]['prev'] = None
        paths[step][item]['probability'] = 1.0 / len(possible_states[step])
    for step in xrange(1, len(given_dist)):
        paths[step] = {}
        for items in paths[step - 1]:
            if items in transition_prob:
                for neighbour in transition_prob[items]:
                    if list(neighbour) in possible_states[step]:
                        if neighbour not in paths[step]:
                            paths[step][neighbour] = {}
                            paths[step][neighbour]['prev'] = items
                            current_proba = paths[step - 1][items]['probability'] * transition_prob[items][neighbour]
                            paths[step][neighbour]['probability'] = current_proba
                        else:
                            current_proba = paths[step - 1][items]['probability'] * transition_prob[items][neighbour]
                            if current_proba > paths[step][neighbour]['probability']:
                                paths[step][neighbour]['prev'] = items
                                paths[step][neighbour]['probability'] = current_proba
    max_p = -1
    final = []
    for c in paths[10]:
        if max_p < paths[step][c]['probability']:
            max_p = paths[step][c]['probability']
            cell = c
    final.append(cell)
    for step in range(10, 0, -1):
        prev = paths[step][cell]['prev']
        final.append(prev)
        cell = prev
    return final


def read_data(filename):
    free, tower_locations, given_dist = [], [], []
    fcrow = 0
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split()
            if len(line) == 0:
                continue
            if len(line) > 5:
                for l in range(len(line)):
                    if line[l] == '1':
                        free.append([fcrow, l])
                fcrow += 1
            if (line[0] == 'Tower') and (len(line) == 4):
                tower_locations.append(tuple([int(line[2]), int(line[3])]))
            elif len(line) == 4:
                given_dist.append([float(ele) for ele in line])
        return free, tower_locations, given_dist


if __name__ == "__main__":
    filename = 'hmm-data.txt'
    free, tower_locations, given_dist = read_data(filename)

    dist_to_towers = []
    for i in range(len(free)):
        dist = []
        for tower in tower_locations:
            euclidean_dist = ((free[i][0] - tower[0]) ** 2 + (free[i][1] - tower[1]) ** 2) ** 0.5
            dist.append([euclidean_dist * 0.7, euclidean_dist * 1.3])
        dist_to_towers.append(dist)

    probable_states_dict = {}
    sdict = {}

    for i in range(0, len(given_dist)):
        probable_states_dict[i] = probable_cells(free, given_dist[i], dist_to_towers)
        for cell in probable_states_dict[i]:
            if tuple(cell) not in sdict:
                sdict[tuple(cell)] = []
            sdict[tuple(cell)].append(i)
    neighbours = {}
    for cell in sdict:
        neighbours[cell] = get_neighbours(cell, 10)
    transition_prob = transition_prob(sdict, neighbours)
    path = viterbi(free, tower_locations, given_dist, dist_to_towers, probable_states_dict, neighbours, transition_prob)
    print("Path:", path[::-1])