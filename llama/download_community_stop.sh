#!/usr/bin/env bash
ps aux | grep 'wget --continue --progress=bar:force https://agi.gpt4.org/llama/LLaMA/' | grep -v grep | awk '{print $2}' | xargs kill
ps aux | grep '.*llama/download_community.sh' | grep -v grep | awk '{print $2}' | xargs kill
