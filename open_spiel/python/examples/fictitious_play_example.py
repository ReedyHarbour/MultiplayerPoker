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

"""Python XFP example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app
from absl import flags

from open_spiel.python.algorithms import exploitability, fictitious_play
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import best_response


import pyspiel
import pickle
FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 10000, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 4, "Number of players")
flags.DEFINE_integer("print_freq", 10, "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game, {"players": FLAGS.players})
  exp = []
  xfp_solver = fictitious_play.XFPSolver(game)
  for i in range(FLAGS.iterations):
    xfp_solver.iteration()
    conv = exploitability.nash_conv(game, xfp_solver.average_policy())
    exp.append(str(conv))
    if i % FLAGS.print_freq == 0:
      print("Iteration: {} Conv: {}".format(i, conv))
      sys.stdout.flush()
  
  with open("fict_4p", "w") as file:
    file.write("\n".join(exp))
  average_policy = xfp_solver.average_policy()
  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * FLAGS.players)
  for i in range(FLAGS.players):
    best_resp_backend = best_response.BestResponsePolicy(
          game, i, average_policy)
    br_policy = [average_policy] * FLAGS.players
    br_policy[i] = best_resp_backend
    br_policy_value = expected_game_score.policy_value(
      game.new_initial_state(), br_policy)
    print("Best response player {} value {}".format(i, br_policy_value[i]))
 #  print("Computed player 0 value: {}".format(average_policy_values[0]))
  # print("Computed player 1 value: {}".format(average_policy_values[1]))
  # print("Computed player 2 value: {}".format(average_policy_values[2]))
  # print("Computed player 3 value: {}".format(average_policy_values[3]))
  
  print("Persisting the model...")
  with open("{}_4_solver.pickle".format("fp"), "wb") as file:
    pickle.dump(xfp_solver.average_policy_tables(), file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  app.run(main)
