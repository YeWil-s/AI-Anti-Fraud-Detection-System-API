import os
import json
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from importlib import import_module


def load_model(model_path, config_path, device):
    with open(config_path, "r") as f:
        full_config = json.load(f)
    model_config = full_config["model_config"]

    module = import_module("models.AASIST")
    model = module.Model(model_config).to(device)

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def preprocess_audio(audio_path, cut=64600):
    try:
        x, sr = librosa.load(audio_path, sr=16000)
        x, _ = librosa.effects.trim(x)

        x_len = x.shape[0]
        if x_len >= cut:
            start = (x_len - cut) // 2
            x_pad = x[start: start + cut]
        else:
            num_repeats = int(cut / x_len) + 1
            x_pad = np.tile(x, (num_repeats))[:cut]

        return torch.from_numpy(x_pad.astype(np.float32)).unsqueeze(0)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def run_inference(input_dir, model, device):
    input_path = Path(input_dir)
    audio_files = list(input_path.glob("*.wav"))

    if not audio_files:
        print(f"No wav files found in {input_dir}")
        return

    results = []
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            x = preprocess_audio(audio_file)
            if x is None: continue

            x = x.to(device)
            _, output = model(x)
            
            # AASIST模型输出: Index 0 = 伪造, Index 1 = 真人
            # prob[0] = 伪造概率, prob[1] = 真人概率
            probs = torch.softmax(output, dim=1)[0]
            fake_prob = probs[0].item()  # 伪造置信度
            real_prob = probs[1].item()  # 真人置信度

            label = "FAKE" if fake_prob > 0.5 else "REAL"
            print(f"[{audio_file.name}] Fake: {fake_prob:.4f}, Real: {real_prob:.4f} -> {label}")

            results.append({
                "file": audio_file.name,
                "fake_score": fake_prob,
                "real_score": real_prob,
                "label": label
            })
    return results


if __name__ == "__main__":
    MODEL_WEIGHTS = "./exp_result/SpeechFake_ZH_AASIST_Subset/weights/best.pth"
    CONFIG_FILE = "./config/AASIST.conf"
    TEST_AUDIO_DIR = "./test/fake"

    if not os.path.exists(TEST_AUDIO_DIR):
        os.makedirs(TEST_AUDIO_DIR)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = load_model(MODEL_WEIGHTS, CONFIG_FILE, device)
            res = run_inference(TEST_AUDIO_DIR, model, device)
            if res:
                real = sum(1 for r in res if r['label'] == "REAL")
                print(f"\nTotal: {len(res)} | Real: {real} | Fake: {len(res) - real}")
        except Exception:
            import traceback

            traceback.print_exc()