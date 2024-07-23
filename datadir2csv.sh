#!/usr/bin/bash

SUFFIX="gt"
DIR=.

while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--help)
			echo "USAGE: $0 [-s (gt|xml)] [ <directory> ]"
			exit
			;;
		-s|--suffix)
			SUFFIX="$2"
			shift
			shift
			;;
		*) 
			DIR="$1"
			shift
			;;
	esac
done

COUNT=0

for i in *.png ; do 
	let COUNT++	
	readlink -nf $i ; 
	echo -ne "\t"; 
	readlink -f ${i%.png}.gt ; 
done

echo "CSV file contains $COUNT entries."

