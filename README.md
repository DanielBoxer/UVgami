![Banner](docs/img/readme/banner.jpg)

![UVgami](docs/img/readme/demo.gif)

## Quickstart

> [!NOTE]
> Check the [user guide](docs/docs.md) for more detailed documentation.

1. Download the latest `UVgami.zip` release [here](https://github.com/danielboxer/UVgami/releases/latest)
2. Drag and drop the zip file into Blender
3. In Blender, press `Download Engine` in the n-panel

There are [three](#engines) supported engines.

![Elephant](docs/img/readme/elephant.jpg)

![Elephant 2](docs/img/readme/elephant_seams.jpg)

![Seam Restrictions](docs/img/readme/elephant_weights.jpg)

![Ostrich](docs/img/readme/ostrich.jpg)

![Rhino](docs/img/readme/rhino.jpg)

## Engines

UVgami has three unwrapping engines, see the [docs](docs/docs.md) for more info:

- OptCuts (CPU): [OptCuts](https://github.com/liminchen/OptCuts) by Minchen Li et al. ([paper](https://www.cs.ubc.ca/labs/imager/tr/2018/OptCuts/))
- xatlas (CPU): [xatlas](https://github.com/jpcy/xatlas) by Jonathan Young
- PartUV (GPU): [PartUV](https://github.com/EricWang12/PartUV) by Zhaoning Wang et al. ([paper](https://arxiv.org/abs/2511.16659)). PartUV AI segmentation can't be used commercially (NVIDIA license)
