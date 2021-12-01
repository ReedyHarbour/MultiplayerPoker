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

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score, best_response
import pyspiel
from open_spiel.python.pytorch import deep_cfr
import pickle
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 10000, "Number of iterations")
flags.DEFINE_integer("num_traversals", 4, "Number of traversals/games")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 3, "Number of players")

def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name, {'players': 3})

  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(32, 32),
      advantage_network_layers=(16, 16),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=None,
      batch_size_strategy=None,
      memory_capacity=int(1e7))

  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)

  average_policy = policy.tabular_policy_from_callable(
      game, deep_cfr_solver.action_probabilities)
  pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
  conv = pyspiel.nash_conv(game, pyspiel_policy)
  logging.info("Deep CFR in '%s' - NashConv: %s", FLAGS.game_name, conv)

  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 3)
  epsilon = 0
  for i in range(FLAGS.players):
    best_resp_backend = best_response.BestResponsePolicy(
          game, i, average_policy)
    br_policy = [average_policy] * FLAGS.players
    br_policy[i] = best_resp_backend
    br_policy_value = expected_game_score.policy_value(
      game.new_initial_state(), br_policy)
    print("Best response player {} value {}".format(i, br_policy_value[i]))
    epsilon = max(br_policy_value[i] - average_policy_values[i], epsilon)
  print("Epsilon: " + str(epsilon))
  print("Persisting the model...")
  with open("{}_solver.pickle".format("deep_cfr"), "wb") as file:
    pickle.dump(deep_cfr_solver, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  app.run(main)
