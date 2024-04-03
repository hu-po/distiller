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

# TODO: ensembles
from llms.api_openai import text as llm

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="/home/oop/dev/data/")
parser.add_argument("--data_dir", type=str, default="/home/oop/dev/data/")
parser.add_argument("--framework", type=str, default="jax")
parser.add_argument("--num_models", type=int, default=2)
parser.add_argument("--num_rounds", type=int, default=2)
parser.add_argument("--cull_ratio", type=int, default=4)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--train_data_dir", type=str, default="sdxl_imagenet_8/train")
parser.add_argument("--test_data_dir", type=str, default="sdxl_imagenet_8/test")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--train_img_mu", type=str, default="0.558373,0.519655,0.478256")
parser.add_argument("--train_img_std", type=str, default="0.207305,0.191163,0.185902")
parser.add_argument("--test_img_mu", type=str, default="0.558373,0.519655,0.478256")
parser.add_argument("--test_img_std", type=str, default="0.207305,0.191163,0.185902")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--early_stop", type=int, default=2)
parser.add_argument("--max_model_size", type=int, default=1e8)
parser.add_argument("--num_tokens", type=int, default=8)
parser.add_argument("--token_dim", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.95)

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

# # Spin up a Tensorboard instance to monitor training
# os.system("pkill -f 'tensorboard'")
# tb_proc = subprocess.Popen(["tensorboard", f"--logdir={logs_dir}"])
# tb_chrome_proc = subprocess.Popen(["/usr/bin/google-chrome", "http://localhost:6006/"])

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
You are a expert machine learning research engineer specializing in {args.framework}.
You are tasked with creating a new model architecture for image encoding.
You will be given several example blocks of code.
Create a new block of code inspired by the given blocks.
Follow any naming conventions in the given blocks and ensure the args and kwargs are the same.
Do not explain, return only the working code which will be written directly to a .py file."""
        user_prompt = ""
        for parent in parents:
            parent_filepath = os.path.join(model_dir, f"{parent}.py")
            with open(parent_filepath, "r") as f:
                user_prompt += f"\n{f.read()}"
        reply = llm(system_prompt + user_prompt)
        reply = llm(
            """
You are an expert debugging machine.
You fix dim mismatch errors in model architectures.
Return the user provided code with any mistakes removed.
Remove any comments.
Do not explain return only the code.""",
            reply,
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
        print(f"Running {args.framework} train for {model}")
        model_filepath = os.path.join(model_dir, f"{model}.py")
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
        train_docker_proc = subprocess.Popen(
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
                f"/src/train.{args.framework}.py",
                f"--seed={args.seed}",
                f"--run_name={model}",
                f"--round={round}",
                f"--save_ckpt={args.save_ckpt}",
                f"--train_data_dir={args.train_data_dir}",
                f"--test_data_dir={args.test_data_dir}",
                f"--img_size={args.img_size}",
                f"--train_img_mu={args.train_img_mu}",
                f"--train_img_std={args.train_img_std}",
                f"--test_img_mu={args.test_img_mu}",
                f"--test_img_std={args.test_img_std}",
                f"--num_epochs={args.num_epochs}",
                f"--batch_size={args.batch_size}",
                f"--early_stop={args.early_stop}",
                f"--max_model_size={args.max_model_size}",
                f"--num_tokens={args.num_tokens}",
                f"--token_dim={args.token_dim}",
                f"--learning_rate={args.learning_rate}",
                f"--b1={args.b1}",
                f"--b2={args.b2}",
            ]
        )
        train_docker_proc.wait()
        if train_docker_proc.returncode != 0:
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