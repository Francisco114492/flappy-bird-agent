# Flappy Bird AI Agent

**Course:** Intelligent Systems II - University of Aveiro

This repository contains the framework for developing and training intelligent agents to play Flappy Bird using **Neuro-evolution**.

## Key Features
* **Backend:** High-concurrency Python server using `asyncio` and `websockets`.
* **Frontend:** Real-time game visualization using HTML5 Canvas.
* **AI Core:** A custom implementation of Genetic Algorithms (GA) and Multi-Layer Perceptrons (MLP) to evolve optimal playing strategies.

## Usage

### Setup

```bash
git clone https://github.com/detiuaveiro/flappy-bird-agent)
cd flappy-bird-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Backend

In a terminal run:

```bash
source venv/bin/activate
python -m src.backend [--pipes] -n <number_of_players> -l <limit>
```

### Training

In a terminal run:

```bash
source venv/bin/activate
python -m src.train -n <number_of_players> -e <number_of_epochs> -a [ga|gwo|egwo|de|pso]
```

### Playing

In a terminal run:

```bash
source venv/bin/activate
python -m src.play -l <model.json>
```

## Documentation

This library was documented using the google style docstring. 
Run the following commands to produce the documentation for this library.

```bash
pdoc --math -d google -o docs src 
```

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
