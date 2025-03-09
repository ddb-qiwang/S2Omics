#!/bin/bash
set -e

prefix=$1

source_img="https://upenn.box.com/s/74bldhvryww9ro76nzhdy1k76ncfrd0i"
source_pixsize="https://upenn.box.com/s/ee0hkjxlr46sa92v88bedsdecqsipw8y"
source_ctname="https://upenn.box.com/s/pw3azq675qkbwmh36tahkmq5n48ji7le"
source_ctdist="https://upenn.box.com/s/cm7g6imljiexmvt3uhrl2oqkez3tvdb8"

target_img="${prefix}he-raw.jpg"
target_pixsize="${prefix}pixel-size-raw.txt"
target_ctname="${prefix}unique_cell_type.pickle"
target_ctdist="${prefix}cell_type_image.pickle"

mkdir -p `dirname $target_img`
wget ${source_img} -O ${target_img}
wget ${source_ctname} -O ${target_ctname}
wget ${source_ctdist} -O ${target_ctdist}
