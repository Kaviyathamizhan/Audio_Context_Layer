import json
import soundfile as sf
from pathlib import Path

scene_ids = ['train_00000', 'train_00250', 'val_00010', 'test_00000', 'test_00050']
for sid in scene_ids:
    split = sid.split('_')[0]
    ann = None
    for line in open(f'dataset_mid/annotations/{split}.jsonl'):
        r = json.loads(line)
        if r['scene_id'] == sid:
            ann = r
            break
    qa_list = [json.loads(line) for line in open(f'dataset_mid/qa/{split}.jsonl') if json.loads(line)['scene_id'] == sid]
    
    audio_path = f"dataset_mid/{ann['file']}"
    info = sf.info(audio_path)
    data, sr = sf.read(audio_path)
    peak = float(abs(data).max())
    
    print('======================================================================')
    print(f"SCENE ID: {sid} ({split}) | Scene Type: {ann['scene']} ({ann['scene_label']})")
    print(f"Audio: {audio_path} | {info.samplerate}Hz | {info.channels}ch | {info.duration:.1f}s | Peak: {peak:.3f}")
    print("Ground-Truth Timeline:")
    for idx, e in enumerate(ann['events']):
        print(f"  [{idx+1}] {e['class']} ({e['label']}): {e['onset']}s -> {e['offset']}s (dur: {e['offset']-e['onset']:.2f}s, src: {e['source']})")
    print(f"Background: {ann['background']['class']} (src: {ann['background']['source']})")
    print(f"QA Pairs ({len(qa_list)} generated):")
    for q in qa_list:
        print(f"  Q ({q['type']}/{q['subtype']}): {q['question']}")
        print(f"    Options: {q['options']} | Answer: {q['answer']}")
