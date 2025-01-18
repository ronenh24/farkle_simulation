import simple_farkle_rl
import simulation


def main() -> None:
    simple_farkle_rl.main()
    simulation.main()

    from utils import max_score
    import os
    import orjson
    from tqdm import tqdm

    if os.path.isfile("max_score.jsonl"):
        max_score_file = open("max_score.jsonl", "wb")
    else:
        max_score_file = open("max_score.jsonl", "xb")
    for dice_combination, result in tqdm(max_score.items()):
        max_score_file.write(orjson.dumps({"dice_combination": dice_combination, "result": result}) + b"\n")
    max_score_file.close()


if __name__ == "__main__":
    main()