#!/usr/bin/env bash

if [ -d "./dist" ] && [ -n "$(ls -A "./dist")" ]; then
  echo "dist folder exists and is not empty"
  exit
fi
python3 -m build && python3 -m twine upload dist/*
