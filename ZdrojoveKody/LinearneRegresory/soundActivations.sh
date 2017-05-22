cd train/


for f in *.wav
do
	# extracting base name from file name
	train_name=""
	if [[ $f =~ (.*[.][0-9]*).wav ]]; then
		train_name=${BASH_REMATCH[1]}
	fi
	
	# getting values for image
	while read p
		do
		file_name=""
		value=""
		if [[ $p =~ (.*)[.]mp4,(.*,.*,.*,.*,*[0-9]) ]]; then
 			file_name=${BASH_REMATCH[1]}
			value=${BASH_REMATCH[2]}
		fi
		if [ "$file_name" == "$train_name" ]
		then
			break
		fi
	done <~/Desktop/BP/first\ impression/values

	echo $f
	python getSoundActivations.py $f $value

done
