# Visualize Results

```bash
usage: graph_student.py [-h] [-m [METHOD]] [-p [PROCESSING]] [-bg [BAGGING]]
                        [-a] [-hs] [-pp] [-c [CAMERA ...]]
                        [-l LEGEND_LOCATION] [-w WIDTH] [-ht HEIGHT] [-sc]
                        [-hl] [-hc] [-cp COLOUR_PALETTE] [-nb]
                        [-i INPUT [INPUT ...]]
                        dataset

positional arguments:
  dataset               ukrface, colormnist, chexpert, fairface

optional arguments:
  -h, --help            show this help message and exit
  -m [METHOD], --method [METHOD]
                        dpsgd, pate (this is fairPATE), vanilla_pate,
                        pate_multidataset, all (default)
  -p [PROCESSING], --processing [PROCESSING]
                        preporcessing, inprocessing, None (default)
  -bg [BAGGING], --bagging [BAGGING]
  -a, --annotate        Annotate the scatter plots. You may want to initially
                        hide all scatterplots with '-hs' as well.
  -hs, --initially_hide_scatter
                        Initially hide scatter plots
  -pp, --paperplot      Create paper plots?
  -c [CAMERA ...], --camera [CAMERA ...]
                        Set camera x y z coordinate: -1.7 2.0 0.75 (default)
  -l LEGEND_LOCATION, --legend_location LEGEND_LOCATION
                        Set legend location: right, top (default)
  -w WIDTH, --width WIDTH
                        Set figure width
  -ht HEIGHT, --height HEIGHT
                        Set figure height
  -sc, --separate_colorbars
                        Show seperate colorbars?
  -hl, --hide_legend    hide legends?
  -hc, --hide_colorbar  hide colorbar?
  -cp COLOUR_PALETTE, --colour_palette COLOUR_PALETTE
                        Define colour palette
  -nb, --no_box         No box for figure? (adjustable width/height)
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Provide a list of input CSV paths (you still need to
                        provide the dataset name)
```

## Example: 
1. Generate paper plot without colorbar or legend (for subplots)
```bash
>> python results/graph_student.py utkface --paperplot -hc -hl
```
2. Generate separate colorbars
```bash
>> python results/graph_student.py utkface --paperplot -sc
```
3. Generate multidataset with seperate colorbars, and legend on the right
```bash
>> python results/graph_student.py utkface,fairface -m pate_multidataset -sc --paperplot -l "right"
```
4. Generate plots from CSVs directly using the input flag `-i` or `--input` (legend on the right, separate colorbars, and no box):
```bash
>> python results/graph_student.py utkface -i results/baselines/utkface_dpsgd_inprocessing.csv results/baselines/utkface_vanilla_pate_inprocessing.csv -l "right" -sc -nb
```
