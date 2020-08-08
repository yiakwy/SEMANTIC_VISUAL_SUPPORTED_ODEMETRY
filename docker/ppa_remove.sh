# credits to original author:

# https://askubuntu.com/questions/65911/how-can-i-fix-a-404-error-when-using-a-ppa-or-updating-my-package-lists
#!/bin/bash
sudo rm /tmp/update.txt; tput setaf 6; echo "Initializing.. Please Wait"
sudo apt-get update >> /tmp/update.txt 2>&1; awk '( /W:/ && /launchpad/ && /404/ ) { print substr($5,26) }' /tmp/update.txt > /tmp/awk.txt; awk -F '/' '{ print $1"/"$2 }' /tmp/awk.txt > /tmp/awk1.txt; sort -u /tmp/awk1.txt > /tmp/awk2.txt
tput sgr0
if [ -s /tmp/awk2.txt ]
then
  tput setaf 1
  printf "PPA's going to be removed\n%s\n" "$(cat /tmp/awk2.txt)"
  tput sgr0
  while read -r line; do echo "sudo add-apt-repository -r ppa:$line"; done < /tmp/awk2.txt > out
  bash out
else
  tput setaf 1
  echo "No PPA's to be removed"
  tput sgr0
fi

