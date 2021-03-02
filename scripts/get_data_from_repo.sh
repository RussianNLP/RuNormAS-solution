#!/bin/bash

data_dir=$1
if [ -d "$data_dir" ]; then
  echo "Data directory $data_dir already exists."
else
  echo "Create data directory $data_dir."
  mkdir "$data_dir"
fi

cd $data_dir

function download_file () {
  local prefix=$1
  local archive_path=$2
  local store_path=$3
  if [ -f "$archive_path" ]; then
    echo "$prefix archive $archive_path already exists."
  else
    echo " $archive_path does not exists. Download $prefix archive."
    wget -O "$archive_path" "https://github.com/dialogue-evaluation/RuNormAS/blob/main/$archive_path?raw=true"
    echo "Unzip $prefix archive"
    unzip "$archive_path" -d "$store_path"
  fi
}

download_file "train" "train_new.zip" "train_new"
download_file "test" "public_test.zip" "public_test"
