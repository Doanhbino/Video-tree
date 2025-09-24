import argparse, os, json, numpy as np
from src.config import settings
from src.video.preprocess import extract_frames
from src.features.clip_extractor import CLIPFeatureExtractor
from src.features.captioner import generate_captions
from src.tree.builder import Leaf, build_adaptive_tree, save_tree

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=settings.TARGET_FPS)
    ap.add_argument("--similarity", type=float, default=settings.SIMILARITY_THRESHOLD)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    frames = extract_frames(args.video, target_fps=args.fps, max_frames=None if args.max_frames == 0 else args.max_frames)
    image_paths = [f.path for f in frames]

    clip = CLIPFeatureExtractor()
    embs = clip.encode_images(image_paths)

    captions = generate_captions(image_paths) if settings.CAPTIONING_ENABLED else ["" for _ in image_paths]
    leaves = [Leaf(idx=i, time_s=float(frames[i].time_s), image_path=image_paths[i], emb=embs[i], caption=captions[i]) for i in range(len(frames))]

    G = build_adaptive_tree(leaves, threshold=args.similarity)
    out = os.path.join(settings.DATA_DIR, settings.TREE_JSON)
    save_tree(G, out)
    print(f"Saved tree to: {out}")

if __name__ == "__main__":
    main()
