#!/bin/bash

dirpath=$1


filename="${dirpath}/commune_ansible/src/mnemonics.txt"

echo "$filename"
#While loop to read line by line
command_prefix=""
mnemonic=""
while IFS= read -r line; do
    #If the line starts with ST then set var to yes.
    if [[ $line == "btcli regen"* ]]; then
        command_prefix="$line"
    else 
        mnemonic="$line"
        eval "$(echo "yes | $command_prefix $mnemonic")"
    fi

done < "$filename"