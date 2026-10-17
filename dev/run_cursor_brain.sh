#!/bin/bash
cd ~/Documents/deepcore-n1
B="$(cat dev/prompts/common.txt)"
T="$(cat dev/prompts/brain.txt)
Second independent attempt: the Codex version already in shared_brain/ is on disk. Write your version under shared_brain/alt/ and do NOT overwrite shared_brain/ files."
cursor-agent -p --trust "$B
$T"
