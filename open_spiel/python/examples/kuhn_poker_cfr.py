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

"""Example use of the CFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import best_response
import pyspiel
import pickle

num_players = 4
def main(_):
  game = pyspiel.load_game("kuhn_poker", {"players": num_players},)

  cfr_solver = cfr.CFRSolver(game)
  iterations = 5000

  for i in range(iterations):
    cfr_solver.evaluate_and_update_policy()
    cfr_value = cfr_solver.average_policy()
    cfr_util = expected_game_score.policy_value(
      game.new_initial_state(), [cfr_value] * num_players)
    print("Game util at iteration {}: {}".format(i, cfr_util))

  average_policy = cfr_solver.average_policy()
  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * num_players)
  for i in range(num_players):
    best_resp_backend = best_response.BestResponsePolicy(
          game, i, average_policy)
    br_policy = [average_policy] * num_players
    br_policy[i] = best_resp_backend
    br_policy_value = expected_game_score.policy_value(
      game.new_initial_state(), br_policy)
    print("Best response player {} value {}".format(i, br_policy_value[i]))
  epsilon = max([br_policy_value[i] - average_policy_values[i] for i in range(num_players)])
  print("Epsilon: " + str(epsilon))
  print("Persisting the model...")
  with open("{}_solver.pickle".format("cfr_py_4"), "wb") as file:
    pickle.dump(cfr_solver, file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  app.run(main)
