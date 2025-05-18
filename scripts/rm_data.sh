#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/files"
  exit 1
fi

path_to_files="$1"

if [ ! -d "$path_to_files" ]; then
  echo "Error: Directory '$path_to_files' does not exist."
  exit 1
fi

total_files=$(ls "$path_to_files" | wc -l)
deleted=0

for file in "$path_to_files"/*; do
    rm "$file"
    deleted=$((deleted + 1))
    tput cr
    echo -n "Deleted $deleted of $total_files files."
done
echo -n "Deleted $deleted of $total_files files."