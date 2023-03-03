#/bin/bash

coldkey=5CFvXBvrfWmK1B44HVguchALhA2Suv4D1pvhwdwAtoD2SNyE
wallet_name=cdg
command_str="btcli regen_coldkeypub --ss58 $coldkey --wallet.name $wallet_name"

eval "$(echo "yes | $command_str")"
