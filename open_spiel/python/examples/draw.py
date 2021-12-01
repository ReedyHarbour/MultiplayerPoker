import matplotlib.pyplot as plt
import os
import numpy as np

def process(str):
    L = str.split(".")
    result = []
    for i in range(1, len(L)):
        if i == 1:
            curr = L[i-1] + "." + L[i][:-1]
        else:
            curr = L[i-1][-1] + "." + L[i][:-1]
        result.append(curr)
        print(curr)
    return result

save_path = "result"
def draw_graph(L):
    all_data = []
    for file in L:
        with open(file, "r") as f:
            if file == "nash_conv_exp_desc_4p":
                processed = process(f.read())
            else:
                processed = f.read().split()
            L1 = [np.log(float(s)) for s in processed]
        all_data.append(L1[:10000])

    plt.rcParams.update({'figure.autolayout': True})

    fig, ax = plt.subplots(nrows=1, ncols=1)
    lab = range(10000)
        # episode_stats = np.array(stats['episode_stats'])
        # steps = np.cumsum(episode_stats[:, 0])
        # reward = episode_stats[:, 1]
    ax.plot(lab, all_data[0], label="CFR-3 players")
    ax.plot(lab, all_data[1], label="CFR-4 players")
    ax.plot(lab, all_data[2], label="CFR-5 players")
    # ax.plot(lab, all_data[3], label="DeepCFR")
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Exploitability')
    ax.legend(loc='best')
    # plt.savefig(os.path.join(save_path, 'reward_train.pdf'))
    plt.savefig(os.path.join(save_path, 'cfr_player_exploitability_log.png'))
    plt.close()

draw_graph(["cfr_3p", "cfr_4p", "cfr_5p"])

def write_new():
    with open("cfr_5p", "r") as f:
        L = f.read().split("\n")
        result = []
        for s in L:
            curr = s.split("exploitability: ")
            print(curr)
            result.append(curr[1])
    
    with open("cfr_5p", "w") as f:
        f.write("\n".join(result))

