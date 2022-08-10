from tqdm import tqdm
import imageio.v2 as imageio
import typer

from pathlib import Path


def run(
    samples_dir: Path,
    out_dir: Path,
    seconds: int = typer.Option(3),
    frame_interval: int = typer.Option(10),
):
    sub_folders = list(samples_dir.glob("*"))
    sub_folders.sort(key=lambda x: int(x.name))
    first = sub_folders[0]

    out_dir.mkdir(exist_ok=True)

    images = list(first.glob("*"))
    image_names = [x.name for x in images]

    for name in tqdm(image_names):
        gif_images = list(samples_dir.glob(f"*/{name}"))
        gif_images.sort(key=lambda x: int(x.parent.name))
        gif_images = gif_images[::frame_interval]
        total_frames = len(gif_images)
        # max FPS is 100 I think??
        fps = min(round(total_frames / seconds), 100)

        files = []
        for file in tqdm(gif_images, leave=False):
            files.append(imageio.imread(file))

        image_name = f"{out_dir}/{name}.gif"
        print(f"Saving image: {image_name} with fps: {fps}\n")
        imageio.mimsave(image_name, files, format="GIF", fps=fps)
        print("Image saved!")


if __name__ == "__main__":
    typer.run(run)
