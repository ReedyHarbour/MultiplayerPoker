# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""MCTS example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import sys
import os
from absl import app
from absl import flags
import numpy as np

import pickle
import itertools
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import best_response
from open_spiel.python.bots import gtp
from open_spiel.python.bots.policy import PolicyBot
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
from open_spiel.python.policy import pyspiel_policy_to_python_policy, FirstActionPolicy, tabular_policy_from_callable, LastActionPolicy
import pyspiel

_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "pass",
    "bet",
    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",

    # Run an alpha_zero checkpoint with MCTS. Uses the specified UCT/sims.
    # Requires the az_path flag.
    "az",
    "cfr",
    "fp", 
    "exp",
    "deep",
]

flags.DEFINE_string("game", "kuhn_poker", "Name of the game.")
flags.DEFINE_enum("player1", "cfr", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "cfr", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_enum("player3", "cfr", _KNOWN_PLAYERS, "Who controls player 3.")
flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", None,
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 1000, "How many simulations to run.")
flags.DEFINE_integer("num_games", 3000, "How many games to play.")
flags.DEFINE_integer("seed", None, "S eed for the random number generator.")
flags.DEFINE_integer("num_players", 3, "Number of players.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_string("solver_path", "cfr_solver.pickle", "Where to find cfr solver.")
FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
  rng = np.random.RandomState(FLAGS.seed)
  if bot_type == "mcts":
    evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "az":
    model = az_model.Model.from_checkpoint(FLAGS.az_path)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if bot_type == "pass":
    curr_policy = get_always_pass_policy(game)
    bot = PolicyBot(player_id, rng, curr_policy)
    return bot
  if bot_type == "bet":
    curr_policy = get_always_bet_policy(game)
    bot = PolicyBot(player_id, rng, curr_policy)
    return bot
  if bot_type == "gtp":
    bot = gtp.GTPBot(game, FLAGS.gtp_path)
    for cmd in FLAGS.gtp_cmd:
      bot.gtp_cmd(cmd)
    return bot
  if bot_type == "exp":
    curr_policy = get_exp_descent_policy(game, "exp_desc_5_solver")
    bot = PolicyBot(player_id, rng, curr_policy)
    return bot
  if bot_type == "cfr":
    curr_policy = get_cfr_policy(game, "cfr_solver_5")
    bot = PolicyBot(player_id, rng, curr_policy)
    return bot
  if bot_type == "deep":
    curr_policy = get_exp_descent_policy(game, "deep_cfr_5_solver")

    bot = PolicyBot(player_id, rng, curr_policy)
    return bot
  if bot_type == "fp":
    curr_policy = get_fp_policy(game, "fp_5_solver")
    bot = PolicyBot(player_id, rng, curr_policy)
    return bot
  raise ValueError("Invalid bot type: %s" % bot_type)

def get_cfr_policy(game, solver):
    print("Loading the model...")
    with open("{}.pickle".format(solver), "rb") as file:
        loaded_solver = pickle.load(file)
    # print("Exploitability of the loaded model: {:.6f}".format(
    #     pyspiel.exploitability(game, loaded_solver.average_policy())))
    curr_policy = loaded_solver.tabular_average_policy()
    # curr_policy = pyspiel.TabularPolicy(game, loaded_solver.average_policy())
    return pyspiel_policy_to_python_policy(game, curr_policy, players=list(range(FLAGS.num_players)))

def get_fp_policy(game, solver):
    with open("{}.pickle".format(solver), "rb") as file:
        loaded_policy = pickle.load(file)
    # print("Exploitability of the loaded model: {:.6f}".format(
    #     pyspiel.exploitability(game, loaded_solver.average_policy())))
    policies = []
    for i in range(FLAGS.num_players):
      policies.append(fictitious_play._callable_tabular_policy(
          loaded_policy[i]))
    joint_policy = fictitious_play.JointPolicy(game, policies)
    return joint_policy
    # return pyspiel_policy_to_python_policy(game, joint_policy, players=[0,1,2])

def get_exp_descent_policy(game, solver):
    with open("{}.pickle".format(solver), "rb") as file:
        loaded_policy = pickle.load(file)
    # print("Exploitability of the loaded model: {:.6f}".format(
    #     pyspiel.exploitability(game, loaded_solver.average_policy())))
    return loaded_policy

def get_always_pass_policy(game):
    policy = FirstActionPolicy(game)
    tab_pol = tabular_policy_from_callable(game, policy, players=None)
    return tab_pol
    # return pyspiel_policy_to_python_policy(game, policy, players=[0,1,2])

def get_always_bet_policy(game):
    policy = LastActionPolicy(game)
    tab_pol = tabular_policy_from_callable(game, policy, players=None)
    return tab_pol

def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None


def _play_game(game, bots, initial_actions):
  """Plays one game."""
  state = game.new_initial_state()
  _opt_print("Initial state:\n{}".format(state))

  history = []

  if FLAGS.random_first:
    assert not initial_actions
    initial_actions = [state.action_to_string(
        state.current_player(), random.choice(state.legal_actions()))]

  for action_str in initial_actions:
    action = _get_action(state, action_str)
    if action is None:
      sys.exit("Invalid action: {}".format(action_str))

    history.append(action_str)
    for bot in bots:
      bot.inform_action(state, state.current_player(), action)
    state.apply_action(action)
    _opt_print("Forced action", action_str)
    _opt_print("Next state:\n{}".format(state))

  while not state.is_terminal():
    current_player = state.current_player()
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      _opt_print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Sampled action: ", action_str)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      # Decision node: sample action for the single current player
      bot = bots[current_player]
      action = bot.step(state)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Player {} sampled action: {}".format(current_player,
                                                       action_str))

    for i, bot in enumerate(bots):
      if i != current_player:
        bot.inform_action(state, current_player, action)
    history.append(action_str)
    state.apply_action(action)

    _opt_print("Next state:\n{}".format(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))

  for bot in bots:
    bot.restart()

  return returns, history

def get_nash_equil(game):
  average_policy = get_fp_policy(game, "fp_3_solver")
  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * FLAGS.num_players)
  for i in range(FLAGS.num_players):
    best_resp_backend = best_response.BestResponsePolicy(
          game, i, average_policy)
    br_policy = [average_policy] * FLAGS.num_players
    br_policy[i] = best_resp_backend
    br_policy_value = expected_game_score.policy_value(
      game.new_initial_state(), br_policy)
    print("Best response player {} value {}".format(i, br_policy_value[i]))
  print("Computed player 0 value: {}".format(average_policy_values[0]))
  print("Computed player 1 value: {}".format(average_policy_values[1]))
  print("Computed player 2 value: {}".format(average_policy_values[2]))
  # print("Computed player 3 value: {}".format(average_policy_values[3]))


def play_game(argv,p1,p2,p3,p4,p5):
  game = pyspiel.load_game(FLAGS.game, {"players": FLAGS.num_players})
  
  bots = [
      _init_bot(p1, game, 0),
      _init_bot(p2, game, 1),
      _init_bot(p3, game, 2),
      _init_bot(p4, game, 3),
      _init_bot(p5, game, 4)
  ]
  histories = collections.defaultdict(int)
  overall_returns = [0] * FLAGS.num_players
  overall_wins = [0] * FLAGS.num_players
  game_num = 0
  try:
    for game_num in range(FLAGS.num_games):
      returns, history = _play_game(game, bots, argv[1:])
      histories[" ".join(history)] += 1
      for i, v in enumerate(returns):
        overall_returns[i] += v
        if v > 0:
          overall_wins[i] += 1
  except (KeyboardInterrupt, EOFError):
    game_num -= 1
    print("Caught a KeyboardInterrupt, stopping early.")
  print("Number of games played:", game_num + 1)
  print("Number of distinct games played:", len(histories))
  print("Players:", p1, p2, p3, p4, p5)
  print("Overall wins", overall_wins)
  print("Overall returns", overall_returns)
  return overall_returns

def get_permutations(players_to_test, k):
  return list(itertools.permutations(players_to_test, k))

def run_all(argv):
  players_to_test = ['cfr','fp','deep','exp','random','pass']
  for (p1,p2,p3,p4,p5) in get_permutations(players_to_test, FLAGS.num_players):
    if 'cfr' not in [p1,p2,p3,p4,p5]:
      continue
    returns = []

    for i in range(6):
      returns.append(play_game(argv,p1,p2,p3,p4,p5))

    mean = np.mean(np.array(returns), axis=0)
    std = np.std(np.array(returns), axis=0)
    output = [p1, p2, p3, p4, p5]
    for i in range(FLAGS.num_players):
      output.append("${:.2f}\pm{:.2f}$".format(mean[i], std[i]))
    with open("output_tournament_5_v2", "a+") as f:
      f.write("& ".join(output) + "\\" + "\\" + "\n")

def _get_nash(argv):
  game = pyspiel.load_game(FLAGS.game, {"players": FLAGS.num_players})
  get_nash_equil(game)
def main(argv):
  play_game(argv, "cfr", "cfr", "cfr")

if __name__ == "__main__":
  app.run(_get_nash)
