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

"""Example use of the MCCFR algorithm on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import pickle
from open_spiel.python.algorithms import expected_game_score, best_response
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "sampling",
    "outcome",
    ["external", "outcome"],
    "Sampling for the MCCFR solver",
)
flags.DEFINE_integer("iterations", 10000, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 4, "Number of players")
flags.DEFINE_integer("print_freq", 1,
                     "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game, {"players": FLAGS.players})
  if FLAGS.sampling == "external":
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
  else:
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)
  for i in range(FLAGS.iterations):
    cfr_solver.iteration()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.nash_conv(game, cfr_solver.average_policy())
      print("Iteration {} exploitability {}".format(i, conv))

  average_policy = cfr_solver.average_policy()
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
  epsilon = max([br_policy_value[i] - average_policy_values[i] for i in range(FLAGS.players)])
  print("Epsilon: " + str(epsilon))
  print("Persisting the model...")
  with open("{}_solver.pickle".format("cfr_py_2"), "wb") as file:
    pickle.dump(cfr_solver, file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
  app.run(main)
