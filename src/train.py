#!/usr/bin/env python


__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import argparse
import asyncio
import enum
import json
import logging
import random
import statistics
import uuid

import numpy as np
import pyBlindOpt.de as de
import pyBlindOpt.egwo as egwo
import pyBlindOpt.ga as ga
import pyBlindOpt.gwo as gwo
import pyBlindOpt.init as init
import pyBlindOpt.pso as pso
from websockets.asyncio.client import connect

import src.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

wslogger = logging.getLogger("websockets")
wslogger.setLevel(logging.WARN)


@enum.unique
class Optimization(enum.Enum):
    """
    Enum data type that represents the optimization algorithm
    """

    de = "de"
    ga = "ga"
    gwo = "gwo"
    pso = "pso"
    egwo = "egwo"

    def __str__(self):
        return self.value


NN_ARCHITECTURE = []


async def play_game(model: nn.NN) -> float:
    """
    Player main loop.

    Args:
        model (nn.NN): the model used for playing

    Returns:
        float: the highscore achieved
    """
    identification = str(uuid.uuid4())[:8]

    uri = f"{CONSOLE_ARGUMENTS.u}/player"

    async with connect(uri) as websocket:
        await websocket.send(json.dumps({"cmd": "join", "id": identification}))

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["evt"] == "world_state":
                # player = data["players"][identification]
                pr = random.random()
                if pr > 0.75:
                    await websocket.send(json.dumps({"cmd": "click"}))
            elif data["evt"] == "done":
                final_fitness = data["highscore"]
                break
    return float(final_fitness)


def objective(p: np.ndarray) -> float:
    """
    Objective function used to evaluate the candidate solution.

    Args:
        p (np.ndarray): the parameters of the candidate solution

    Returns:
        float: the cost value
    """
    model = nn.NN(NN_ARCHITECTURE)
    model.update(p)

    highscore = asyncio.run(play_game(model))
    return -highscore


async def share_training_data(epoch: int, obj: list) -> None:
    """
    Method that sends the training data to the viewer.

    Args:
        epoch (int): the current epoch
        obj (list): list with the current objective values
    """
    # compute the worst, best and average fitness
    worst = max(obj)
    best = min(obj)
    mean = statistics.mean(obj)

    uri = f"{CONSOLE_ARGUMENTS.u}/training"

    async with connect(uri) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "cmd": "training",
                    "epoch": epoch,
                    "worst": worst,
                    "best": best,
                    "mean": mean,
                }
            )
        )


def callback(epoch: int, obj: list, pop: list) -> None:
    """
    Callback used to share the training data to the viewer.

    Args:
        epoch (int): the current epoch
        obj (list): list with the current objective values
    """
    asyncio.run(share_training_data(epoch, obj))


def store_data(model: list, parameters: np.ndarray, path: str) -> None:
    """
    Store the model into a json file.

    Args:
        model (list): the model definition
        parameters (np.ndarray): the model parameters
        path (str): the location of the file
    """
    with open(path, "w") as f:
        json.dump({"model": model, "parameters": parameters.tolist()}, f)


def main(args: argparse.Namespace) -> None:
    """
    Main method.

    Args:
        args (argparse.Namespace): the program arguments
    """
    # Define the random seed
    np.random.seed(args.s)

    # Define the bounds for the optimization
    bounds = np.asarray([[-1.0, 1.0]] * nn.network_size(NN_ARCHITECTURE))

    # Generate the initial population
    population = [nn.NN(NN_ARCHITECTURE).ravel() for i in range(args.n)]

    # Apply Opposition Learning to the inital population
    # population = init.opposition_based(objective, bounds, population=population, n_jobs=args.n)
    # population = init.round_init(objective, bounds, n_pop=args.n, n_rounds=5, n_jobs=args.n)
    population = init.oblesa(objective, bounds=bounds, n_pop=args.n, n_jobs=args.n)

    # Run the optimization algorithm
    if args.a is Optimization.de:
        best, _ = de.differential_evolution(
            objective,
            bounds,
            variant="best/3/exp",
            callback=callback,
            population=population,
            n_iter=args.e,
            n_jobs=args.n,
            cached=False,
            verbose=True,
            seed=args.s,
        )
    elif args.a is Optimization.ga:
        best, _ = ga.genetic_algorithm(
            objective,
            bounds,
            n_iter=args.e,
            callback=callback,
            population=population,
            n_jobs=args.n,
            cached=False,
            verbose=True,
            seed=args.s,
        )
    elif args.a is Optimization.pso:
        best, _ = pso.particle_swarm_optimization(
            objective,
            bounds,
            n_iter=args.e,
            callback=callback,
            population=population,
            n_jobs=args.n,
            cached=False,
            verbose=True,
            seed=args.s,
        )
    elif args.a is Optimization.gwo:
        best, _ = gwo.grey_wolf_optimization(
            objective,
            bounds,
            n_iter=args.e,
            callback=callback,
            population=population,
            n_jobs=args.n,
            cached=False,
            verbose=True,
            seed=args.s,
        )
    elif args.a is Optimization.egwo:
        best, _ = egwo.enhanced_grey_wolf_optimization(
            objective,
            bounds,
            n_iter=args.e,
            callback=callback,
            population=population,
            n_jobs=args.n,
            cached=False,
            verbose=True,
            seed=args.s,
        )
    else:
        raise ValueError(f"Optimization algorithm '{args.a}' is not implemented.")

    # store the best model
    store_data(NN_ARCHITECTURE, best, args.o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-u", type=str, help="server url", default="ws://localhost:8765"
    )
    parser.add_argument("-s", type=int, help="Random generator seed", default=42)
    parser.add_argument("-e", type=int, help="optimization epochs", default=30)
    parser.add_argument("-n", type=int, help="population size", default=10)
    parser.add_argument(
        "-a",
        type=Optimization,
        help="Optimization algorithm",
        choices=list(Optimization),
        default="pso",
    )
    parser.add_argument(
        "-o", type=str, help="store the best model", default="out/model.json"
    )
    args = parser.parse_args()
    # Global variable for the arguments (required due to the callback function)
    CONSOLE_ARGUMENTS = args

    main(args)
