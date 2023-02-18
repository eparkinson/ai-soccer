import os
import neat
from multiprocessing import Pool

from aisoccer.brains.BehindAndTowards import BehindAndTowards
from aisoccer.brains.DefendersAndAttackers import DefendersAndAttackers
from aisoccer.brains.RandomWalk import RandomWalk
from aisoccer.brains.NNBrain import NNBrain
from aisoccer.tournament import *

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    genome_list = []
    fitness_map = {}
    
    for genome_id, genome in genomes:
        genome_list.append((genome_id, genome, config))
    
    with Pool(16) as pool:
        result = pool.map(eval_genome, genome_list)

    for genome_id, fitness in result:        
        fitness_map[genome_id] = fitness

    for genome_id, genome in genomes:
        genome.fitness = fitness_map[genome_id]


def eval_genome(g):
    net = neat.nn.FeedForwardNetwork.create(g[1], g[2])    
    g[1].fitness = eval_tournament(g[0], net)
    print("*** Genome: " + str(g[0]) + " Fitness: " + str(g[1].fitness))
    return g[0], g[1].fitness

def eval_tournament(genome_id, net):
    fitness = 0

    nnbrain = NNBrain("GENOME-"+str(genome_id))
    nnbrain.set_net(net)

    batbrain = BehindAndTowards("BAT")
    daabrain = DefendersAndAttackers("DAA")

    batgame = Game(nnbrain, batbrain, 3000, True)
    daagame = Game(nnbrain, daabrain, 3000, True)
    batscore = batgame.play()
    daascore = daagame.play()
    
    
    fitness =  batscore['blue'] - batscore['red'] + daascore['blue'] - daascore['red']


    return fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config\\neat_demo.cfg')
    print(config_path)
    run(config_path)
