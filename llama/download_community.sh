#!/usr/bin/env bash
set -euo pipefail

PRESIGNED_URL="https://agi.gpt4.org/llama/LLaMA"
ALL_MODELS=7B,13B,30B,65B

YELLOW=$(tput setaf 3)
RED=$(tput setaf 1)
CLEAR=$(tput sgr0)

function usage {
    cat <<EOF
Usage: download_community [-vh] [<models>] [<output_directory>]

Download the given llama <models> to <output_directory>. By default, will
download all available models into the current directory

OPTIONS

  -v, --verbose: enable verbose mode
  -h, --help:    print this help and exit

EXAMPLES

Download all models ($ALL_MODELS) into the current directory

  ./download_community.sh

Download the 7B and 13B parameter models to /usr/share/llama
    
  ./download_community.sh 7B,13B /usr/share/llama

EOF
    exit 1
}

# print its argument in red and quit 
function die {
    printf "%s%s%s\n" "$RED" "$1" "$CLEAR"
    exit 1
}

# print its argument in yellow
function log {
    printf "\n%s%s%s\n" "$YELLOW" "$1" "$CLEAR"
}

# download a file with a progress bar, then display a success message. Takes
# two arguments: the URL and the output file name
function download {
    if ! wget --continue --progress=bar:force "$1" -O "$2"; then
        die "failed to download $1 -> $2"
    fi
    echo ✅ "$2"
}

# change into the model directory and use md5sum -c to verify the checksums of
# the model files within. Uses a subshell to avoid changing the script's
# direcotry
function verify {
    (cd "$1" && md5sum -c "$2")
}

# return the number of shards for a given model. Bash 3 doesn't support
# associative arrays, so use a case statement instead.
function nshards {
    case $1 in
        7B)
            echo 0
            ;;
        13B)
            echo 1
            ;;
        30B)
            echo 3
            ;;
        65B)
            echo 7
            ;;
        *)
            die "invalid argument to nshards: $1"
            ;;
    esac

}

# check for wget - if it's not present print an error
if ! command -v wget &> /dev/null
then
    die "wget not found. You must have wget installed and on your path to run this script"
fi

# parse the optional flags and discard them
while true; do
    case $1 in
        -v|--verbose)
            set -x
            shift
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            break
            ;;
    esac
done

# MODELS_TO_DOWNLOAD is a comma-separated list of models the user wants to
# download, which defaults to all models. Split it into an array called MODELS
MODELS_TO_DOWNLOAD=${1:-$ALL_MODELS}
IFS="," read -r -a MODELS <<< "$MODELS_TO_DOWNLOAD"

# TARGET_FOLDER is the root directory to download the models to
TARGET_FOLDER=${2:-.}

log "❤️  Resume download is supported. You can ctrl-c and rerun the program to resume the downloading"

# ensure the targeted directory exists
mkdir -p "$TARGET_FOLDER"

log "Downloading tokenizer..."
download "$PRESIGNED_URL/tokenizer.model" "$TARGET_FOLDER/tokenizer.model"
download "$PRESIGNED_URL/tokenizer_checklist.chk" "$TARGET_FOLDER/tokenizer_checklist.chk"
verify "$TARGET_FOLDER" tokenizer_checklist.chk

# for each model, download each of its shards and then verify the checksums
for model in "${MODELS[@]}"
do
    log "Downloading $model"
    mkdir -p "$TARGET_FOLDER/$model"

    # download each shard in the model
    for s in $(seq -f "0%g" 0 "$(nshards "$model")")
    do
       fout="$TARGET_FOLDER/$model/consolidated.$s.pth"
       log "downloading file to $fout ...please wait for a few minutes ..."
       download "$PRESIGNED_URL/$model/consolidated.$s.pth" "$fout"
    done

    # download the params and checksums
    download "$PRESIGNED_URL/$model/params.json" "$TARGET_FOLDER/$model/params.json"
    download "$PRESIGNED_URL/$model/checklist.chk" "$TARGET_FOLDER/$model/checklist.chk"

    log "Checking checksums for the $model model"
    verify "$TARGET_FOLDER/$model" checklist.chk
done
