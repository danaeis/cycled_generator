#!/bin/bash

echo "Downloading models/search_code_18590.pt ..."
gdown --id 1G2YZo1HKkvf5sMvaaRQeZp_YA9GDlvL- -O bundles/multi_organ_segmentation/models/search_code_18590.pt
md5sum bundles/multi_organ_segmentation/models/search_code_18590.pt

echo "Downloading models/model.pt ..."
gdown --id 1kH0yTyiXUNqdYXpnSXI2p5-vFwDYoCzl -O bundles/multi_organ_segmentation/models/model.pt
md5sum bundles/multi_organ_segmentation/models/model.pt

echo "Downloading models/model.ts ..."
gdown --id 1qwV99IfYvLzpjsHgfrlciqDp8yQ5uLXs -O bundles/multi_organ_segmentation/models/model.ts
md5sum bundles/multi_organ_segmentation/models/model.ts
