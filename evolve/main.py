import argparse
import base64
import glob
import os
import random
import requests
import shutil
import subprocess
import time
import uuid
import yaml

from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# from llms import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data/")
parser.add_argument("--data_dir", type=str, default="/home/oop/dev/data/")
# parser.add_argument("--framework", type=str, default="pytorch")
parser.add_argument("--framework", type=str, default="jax")
parser.add_argument("--num_models", type=int, default=2)
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--cull_ratio", type=int, default=4)
args = parser.parse_args()

print("🧙‍♂️ Starting Evolution")
random.seed(args.seed)
session_id = str(uuid.uuid4())[:6]
base_dir = os.path.join(args.base_dir, f"evolve.{session_id}")
os.makedirs(base_dir, exist_ok=True)
print(f"base directory at {base_dir}")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)
print(f"model directory at {model_dir}")
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
print(f"logs directory at {logs_dir}")
ckpt_dir = os.path.join(base_dir, "ckpt")
os.makedirs(ckpt_dir, exist_ok=True)
print(f"ckpt directory at {ckpt_dir}")
data_dir = args.data_dir
assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"

# Spin up a Tensorboard instance to monitor training
os.system("pkill -f 'tensorboard'")
tb_proc = subprocess.Popen(["tensorboard", f"--logdir={logs_dir}"])
tb_chrome_proc = subprocess.Popen(["/usr/bin/google-chrome", "http://localhost:6006/"])

# Build and update the docker container for evolution
build_docker_proc = subprocess.Popen(
    [
        "docker",
        "build",
        "-t",
        f"evolve.{args.framework}",
        "-f",
        f"Dockerfile.{args.framework}",
        ".",
    ]
)
build_docker_proc.wait()
assert build_docker_proc.returncode == 0, "Error building docker container"

# Seed with models from the model directory
seed_models_dir = os.path.join(os.getcwd(), "models", args.framework)
models = os.listdir(seed_models_dir)
for model in models:
    shutil.copy(os.path.join(seed_models_dir, model), model_dir)
    print(f"Evolution seeded with model {model}")

# Remove the player suffix from the player names
models = [x.split(".")[0] for x in models]
# shuffle the players and clip to num_models
random.shuffle(models)
models = models[: args.num_models]

# evolution runs as a series of rounds
for round in range(args.num_rounds):
    print(f"Starting evolution rounds {round}")
    # reproduce to fill in missing players
    while len(models) < args.num_models:
        run_id = str(uuid.uuid4())[:6]
        print(f"Creating run {run_id}")
        parents = random.sample(models, 2)
        print(f"Reproducing {parents[0]} and {parents[1]}")
        # Add parent names to run_id for easy identification
        run_id = f"{parents[0][:2]}_{parents[1][:2]}_{run_id}"
        # zero-shot
        system_prompt = f"""
You are a expert machine learning research engineer.
You excel at creating new and unique model architectures.
You use {args.framework} and make use of the einops library.
You will be given several example blocks of code.
Create a new block of code inspired by the given blocks.
The block of code should be called `Block` and should be a subclass of `nn.Module`.
Make sure the kwarg `num_classes` is present in the `__init__` method.
Do not explain, return only the working code which will be written directly to a .py file."""
        user_prompt = ""
        for parent in parents:
            parent_filepath = os.path.join(model_dir, f"{parent}.py")
            with open(parent_filepath, "r") as f:
                user_prompt += f"\n{f.read()}"
        reply = llm(system_prompt, user_prompt, 0.9, 512)
        reply = llm(
            """
You are an expert debugging machine.
You fix dim mismatch errors in model architectures.
Return the user provided code with any mistakes removed.
Remove any comments.
Do not explain return only the code.""",
            reply,
            0.7,
            512,
        )
        run_filename = f"{run_id}.py"
        run_filepath = os.path.join(model_dir, run_filename)
        with open(run_filepath, "w") as f:
            # HACK: removes first and last lines
            f.write("\n".join(reply.split("\n")[1:-1]))
        models.append(run_id)

    best_scores = {}
    results_filepath = os.path.join(ckpt_dir, f"results.r{round}.yaml")
    with open(results_filepath, "w") as f:
        yaml.dump({}, f)
    previous_results_filepath = os.path.join(ckpt_dir, f"results.r{round-1}.yaml")
    if os.path.exists(previous_results_filepath):
        with open(previous_results_filepath, "r") as f:
            previous_results = yaml.safe_load(f)
    else:
        previous_results = {}
    for model in models:
        # skip any models that have already been evaluated
        if model in previous_results:
            print(f"Skipping {model} as it has already been evaluated")
            best_scores[model] = previous_results[model]["test_accuracy"]
            continue
        print(f"Running {args.framework} traineval for {model}")
        model_filepath = os.path.join(model_dir, f"{model}.py")
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
        traineval_docker_proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--gpus=all",
                "-v",
                f"{model_filepath}:/src/model.py",
                "-v",
                f"{ckpt_dir}:/ckpt",
                "-v",
                f"{logs_dir}:/logs",
                "-v",
                f"{data_dir}:/data",
                f"evolve.{args.framework}",
                "python",
                f"/src/traineval.{args.framework}.py",
                f"--run_name={model}",
                f"--round={round}",
                f"--num_epochs={1}",
                f"--batch_size={1}",
                f"--early_stop={1}",
            ]
        )
        traineval_docker_proc.wait()
        if traineval_docker_proc.returncode != 0:
            print(f"Error occurred when training model {model}")
            best_scores[model] = 0.0
        else:
            print(f"Trained model {model}")
            with open(results_filepath, "r") as f:
                player_results = yaml.safe_load(f)
            best_scores[model] = player_results[model]["test_accuracy"]
        print(f"Model {model} result {best_scores[model]}")

    # Sort the models by scores, cull the bottom models
    sorted_models = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted models: {sorted_models}")
    cull_index = len(sorted_models) // args.cull_ratio
    best_models = [x[0] for x in sorted_models[:cull_index]]
    print(f"Best models: {best_models}")
    worst_models = [x[0] for x in sorted_models[-cull_index:]]
    print(f"Worst models: {worst_models}")
    for model in worst_models:
        os.remove(os.path.join(model_dir, f"{model}.py"))
        print(f"Removed model {model}")
    models = [x for x in models if x not in worst_models]

    # Plot round results
    plot_filepath = os.path.join(ckpt_dir, "test_accuracy_plot.png")
    yaml_files = glob.glob(f"{ckpt_dir}/results.r*.yaml")
    rounds = []
    test_acc = []
    for file in yaml_files:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        round_number = int(file.split(".")[-2].split("r")[-1])
        for key in data:
            rounds.append(round_number)
            test_acc.append(data[key]["test_accuracy"])

    plt.scatter(rounds, test_acc)
    plt.xlabel("round")
    plt.ylabel("acc")
    plt.title("evolution")
    plt.xlim(0, 32)
    plt.ylim(0, 1)
    plt.savefig(plot_filepath)