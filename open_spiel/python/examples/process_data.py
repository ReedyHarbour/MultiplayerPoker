import os 
import matplotlib.pyplot as plt
from kuhn_tournament import get_permutations
import numpy as np
file_name = "output_tournament_v2"
num_players = 3
def process(L):
    output = []
    for x in L:
        x = x.strip()
        x = x.strip("$")
        mean,std = x.split("\pm")
        output.append((float(mean),float(std)))
    return output

def draw_graph(L,std):
    plt.rcParams.update({'figure.autolayout': True})

    fig, ax = plt.subplots(nrows=1, ncols=1)
    lab = [1,2,3,4]
        # episode_stats = np.array(stats['episode_stats'])
        # steps = np.cumsum(episode_stats[:, 0])
        # reward = episode_stats[:, 1]
    for algo in L:
        ax.plot(lab, L[algo], marker='o', label=algo)
        # ax.fill_between(lab, np.array(L[algo]) - np.array(std[algo]), np.array(L[algo]) + np.array(std[algo]), alpha=0.3)
    # ax.plot(lab, all_data[3], label="DeepCFR")
    ax.set_xlabel('Player Position')
    ax.set_ylabel('Payoff')
    ax.legend(loc="lower left")
    # plt.savefig(os.path.join(save_path, 'reward_train.pdf'))
    plt.savefig(os.path.join("result", 'all_player_4.png'))
    plt.close()

def rename():
    with open(file_name, "r") as f:
        content = f.read().split("\n")
    all = ['cfr','fp','deep','exp','random','pass','bet']
    i = 0
    for player_combo in get_permutations(all, num_players):
        if 'deep' not in player_combo:
            continue
        curr = content[i]
        curr_L = curr.split("& ")
        if len(curr_L) == 7:
            curr_L = list(player_combo) + curr_L[3:]
            content[i] = "& ".join(curr_L)
        i += 1
    with open("output_tournament_4_v3", "w+") as f:
        f.write("\n".join(content))

def get_mean():
    with open(file_name, "r") as f:
        content = f.read()

    content_by_line = content.split("\n")
    mean_by_player = [dict() for i in range(num_players)]
    std_by_player = [dict() for i in range(num_players) ]
    for line in content_by_line:
        line = line.strip("\\\\")
        if line == "":
            continue
        res_line = line.split("& ")
        player_pos = res_line[:num_players]
        output = process(res_line[num_players:])
        for pos in range(num_players):
            mean_by_player[pos][player_pos[pos]] = mean_by_player[pos].get(player_pos[pos], 0)
            mean_by_player[pos][player_pos[pos]] += output[pos][0]
            std_by_player[pos][player_pos[pos]] = std_by_player[pos].get(player_pos[pos], 0)
            std_by_player[pos][player_pos[pos]] += output[pos][1]

    output_L = []
    L = dict()
    std = dict()
    PERM = 30
    for pos in range(num_players):
        for player in mean_by_player[pos]:
            L[player] = L.get(player, [])
            L[player].append(mean_by_player[pos][player]/PERM)
            std[player] = std.get(player, [])
            std[player].append(std_by_player[pos][player]/PERM)
            s ="{} & {} & {:.4f} \pm {:.4f}\\\\".format(pos+1, player, mean_by_player[pos][player]/PERM, std_by_player[pos][player]/PERM)
            # output_L.append(s)
            print(s)

    for player in L:
        s = "{} & ${:.4f}\pm{:.4f}$\\\\".format(player, sum(L[player]) / num_players, sum(std[player]) / num_players)
        output_L.append(s)

    # draw_graph(L,std)

    with open("processed_tournament_3", "a+") as f:
        f.write("\n".join(output_L))

get_mean()