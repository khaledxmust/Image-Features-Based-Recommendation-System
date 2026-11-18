"""Generate an original textured test pair without shipping catalogue photographs."""

from pathlib import Path
import random

from PIL import Image, ImageDraw


def main():
    output = Path(__file__).parent / "generated"
    output.mkdir(exist_ok=True)
    image = Image.new("RGB", (640, 480), "#ece7de")
    draw = ImageDraw.Draw(image)
    randomizer = random.Random(2026)
    for _ in range(200):
        x, y = randomizer.randrange(20, 610), randomizer.randrange(20, 450)
        radius = randomizer.randrange(3, 18)
        color = tuple(randomizer.randrange(20, 200) for _ in range(3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.polygon([(200, 160), (260, 90), (380, 90), (440, 160), (400, 240), (380, 220), (380, 390), (260, 390), (260, 220), (240, 240)], fill="#273e69", outline="#10151d", width=4)
    draw.line((320, 95, 320, 385), fill="#d8b77c", width=5)
    for y in range(160, 360, 40):
        draw.ellipse((311, y, 329, y + 18), fill="#d8b77c")
    image.save(output / "image1.png")
    image.rotate(7, resample=Image.Resampling.BICUBIC, translate=(10, -8), fillcolor="#ece7de").save(output / "image2.png")
    print(f"Generated test images in {output}")


if __name__ == "__main__":
    main()
