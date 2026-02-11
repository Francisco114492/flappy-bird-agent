#!/usr/bin/env python


__author__ = "Mário Antunes"
__version__ = "0.1"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import argparse
import asyncio
import json
import logging
import random
import uuid

import numpy as np
import websockets

import src.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

wslogger = logging.getLogger("websockets")
wslogger.setLevel(logging.WARN)


def load_data(path: str) -> tuple:
    """
    Load a json encoded model.

    Args:
        path(str): the location of the model

    Returns:
        tuple: model definition and parameters
    """
    with open(path, "rb") as f:
        data = json.load(f)
        return data["model"], np.asarray(data["parameters"])


async def play_game(url: str, model: nn.NN | None) -> float:
    """
    Player main loop.

    Receives the world dump, and decides if it click of not.
    Always shares the network information.

    Args:
        url (str): the server url
        model (nn.NN): the NN model

    Returns:
        float: highscore
    """
    identification = str(uuid.uuid4())[:8]
    final_score = 0.0
    async with websockets.connect(f"{url}/player") as websocket:
        await websocket.send(json.dumps({"cmd": "join", "id": identification}))
        done = False
        while not done:
            data = json.loads(await websocket.recv())
            if data["evt"] == "world_state":
                # player = data["players"][identification]
                pr = random.random()
                if pr > 0.75:
                    await websocket.send(json.dumps({"cmd": "click"}))
            elif data["evt"] == "done":
                done = True
                final_score = data["highscore"]
    return final_score


def main(args: argparse.Namespace) -> None:
    """
    Main method.

    Args:
        args (argparse.Namespace): the program arguments
    """
    # model_description, parameters = load_data(args.l)
    # model = nn.NN(model_description)
    # model.update(parameters)
    model = None

    highscore = asyncio.run(play_game(args.u, model))
    logger.info(f"Highscore: {highscore}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play the game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-u", type=str, help="server url", default="ws://localhost:8765"
    )
    parser.add_argument(
        "-l", type=str, help="load a player neural network", required=True
    )
    args = parser.parse_args()

    main(args)
