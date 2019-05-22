from __future__ import print_function
import os
import neat
import itertools
from random import random as rand
from time import sleep

n = 100

ins = []
outs = []
for x in range(n):
    t = [round(rand(), 4), round(rand(), 4)]
    ins.append(t)
    outs.append(t[1] >= t[0]**2)


def fit(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = n
        for i, o in zip(ins, outs):
            genome.fitness -= abs(net.activate(i)[0]-o)**2      
        genome.fitness /= n


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # Run for up to 300 generations.
    winner = p.run(fit, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    c = 0.0
    for i, o in zip(ins, outs):
        output = round(winner_net.activate(i)[0], 2)
        if output == round(o, 2): c+=1.0
        print("input {!r}, expected output {!r}, got {!r}".format(i, round(o, 2), output))
    print ("Number of correct results:", c, "Percent of results correct: "+str(c/float(n)*100.0))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
